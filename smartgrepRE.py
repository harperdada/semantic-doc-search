#!/usr/bin/env python3
"""
smartgrep.py  (Real Estate / DRE / Contracts RAG over ~/Downloads/*.pdf)

- Builds/loads a persisted FAISS index (separate from your technical index)
- Indexes ONLY PDFs whose filename matches real-estate / legal-ish keywords
- Answers questions using ONLY retrieved excerpts and cites sources

Usage:
  python3 smartgrep.py
  python3 smartgrep.py --rebuild

Tuning:
  export DEBUG_SCORES=1
  export LOW_CONF_THRESHOLD=0.95
"""

import glob
import os
import signal
import argparse
from textwrap import indent
import re
from typing import List, Tuple, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pypdf import PdfReader


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DOCUMENT_GLOB = os.path.expanduser("~/Downloads/*.pdf")

# WHITELIST: only PDFs whose *filename* contains one of these (case-insensitive) are indexed.
# Add/remove based on your actual filenames (DRE courses, RE forms, disclosures, contracts, etc.)
WHITELIST_KEYWORDS = [
    # Exams / DRE / courses
    "real estate", "real_estate", "dre", "caldre", "california real estate",
    "principles", "practice", "broker", "salesperson", "license", "licensing",
    "agency", "fiduciary", "disclosure", "transfer", "statement",
    "re 400a", "re400a", "re 435", "re435", "re 419", "re419",

    # Forms / contracts / escrow / title
    "purchase", "agreement", "contract", "offer", "counteroffer",
    "escrow", "title", "alta", "settlement", "closing", "hud", "cd",
    "addendum", "amendment", "contingency", "inspection", "appraisal",

    # Ownership / deeds / property docs
    "deed", "grant deed", "quitclaim", "trust", "hoa", "covenant", "cc&r", "ccr",
    "lease", "tenant", "landlord", "rent", "eviction", "notice",

    # Finance / lending
    "mortgage", "loan", "promissory", "note", "apr", "rate", "lender",
    "underwriting", "closing disclosure",

    # Common legal-ish terms that appear in file names
    "policy", "terms", "conditions", "regulation", "code", "act",
]

# Optional: always skip these
IGNORE_ALWAYS = [
    # keep your old ignore if you want; add more if needed
    "w2", "1099", "taxreturn", "tax return",
]

PDF_LOAD_TIMEOUT_SECONDS = 900  # real-estate docs can be big scans; 15 min is fine for rebuilds

# Persisted FAISS index (separate from your technical one)
INDEX_DIR = "faiss_index_realestate"

# Retrieval settings
RETRIEVE_K = 12          # pull more, then rerank
CONTEXT_K = 5            # feed best N to the LLM

# (Keep for debugging; note: FAISS distance is NOT a perfect "junk" detector)
LOW_CONF_THRESHOLD = float(os.environ.get("LOW_CONF_THRESHOLD", "0.95"))
DEBUG_SCORES = os.environ.get("DEBUG_SCORES", "1").lower() in ("1", "true", "yes")

# Source bias (prefer these in answers if present)
SOURCE_PRIORITY_PATTERNS = [
    "california",
    "dre",
    "real estate principles",
    "real estate practice",
    "agency",
    "alta",
    "settlement",
    "purchase agreement",
    "disclosure",
    "boe-64",
    "san joaquin",
]

# Domain signals (used for reranking)
LEGAL_SIGNALS = [
    "shall", "hereby", "whereas", "pursuant", "subject to", "notwithstanding",
    "liability", "indemnify", "indemnification", "warranty", "representations",
    "breach", "default", "termination", "remedy", "damages", "penalty",
    "governing law", "jurisdiction", "venue", "arbitration", "mediation",
    "assignment", "severability", "entire agreement", "force majeure",
    "section", "clause", "article", "exhibit", "addendum", "appendix",
]

RE_SIGNALS = [
     "crops", "crop", "harvest", "harvested",
     "emblements", "severance", "severed",
     "fructus industriales", "fructus naturales",
     "annual crops", "growing crops",
     "fixtures", "annexation", "attached",
     "personal property", "real property",
    "agency", "fiduciary", "disclosure", "transfer disclosure statement",
    "tds", "seller", "buyer", "broker", "agent", "listing", "commission",
    "escrow", "title", "closing", "settlement", "contingency",
    "inspection", "appraisal", "earnest money", "deposit",
    "hoa", "covenant", "cc&r", "condominium", "pud",
    "deed", "grant deed", "quitclaim", "trust deed",
    "mortgage", "note", "promissory", "lender", "loan",
    "fair housing", "redlining", "steering",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_whitelisted(path: str) -> bool:
    name = os.path.basename(path).lower()

    if any(k in name for k in IGNORE_ALWAYS):
        return False

    return any(k in name for k in WHITELIST_KEYWORDS)


def is_encrypted_pdf(path: str) -> bool:
    try:
        r = PdfReader(path)
        return bool(getattr(r, "is_encrypted", False))
    except Exception:
        return False


class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("PDF load timed out")


def load_pdf_with_timeout(path: str, timeout_s: int):
    # Allow disabling timeouts by setting to 0 or negative
    if timeout_s <= 0:
        loader = PyPDFLoader(path)
        return loader.load()

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        loader = PyPDFLoader(path)
        return loader.load()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _is_priority_source(src: str) -> bool:
    s = (src or "").lower()
    return any(p in s for p in SOURCE_PRIORITY_PATTERNS)


def _hits(text: str, signals: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for s in signals if s in t)


def _term_overlap(query: str, text: str) -> int:
    q_terms = set(re.findall(r"[a-z0-9\.\-]+", (query or "").lower()))
    t_terms = set(re.findall(r"[a-z0-9\.\-]+", (text or "").lower()))
    if not q_terms or not t_terms:
        return 0
    return len(q_terms.intersection(t_terms))


def _rerank(query: str, scored_docs: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
    """
    Takes similarity_search_with_score output: [(doc, score), ...]
    Returns list of dicts: {"doc": doc, "score": score, "_rerank": ..., ...}
    """
    out: List[Dict[str, Any]] = []
    for doc, score in scored_docs:
        src = doc.metadata.get("source", "")
        txt = doc.page_content or ""

        pri = 6 if _is_priority_source(src) else 0
        legal = _hits(txt, LEGAL_SIGNALS)
        re_sig = _hits(txt, RE_SIGNALS)
        ov = _term_overlap(query, txt)

        # Rerank score: prioritize authoritative sources + legal/RE signals + overlap
        rer = pri + (1.0 * legal) + (1.0 * re_sig) + (0.25 * ov)

        out.append({
            "doc": doc,
            "score": float(score),
            "_rerank": float(rer),
            "_legal_hits": int(legal),
            "_re_hits": int(re_sig),
            "_src_priority": bool(pri),
        })

    out.sort(key=lambda r: (r["_rerank"], -abs(r["score"])), reverse=True)
    return out


def _format_context(ranked: List[Dict[str, Any]], k: int) -> Tuple[str, List[Dict[str, Any]]]:
    used = ranked[:k]
    blocks = []
    for i, r in enumerate(used, start=1):
        d = r["doc"]
        src = d.metadata.get("source", "<unknown>")
        page = d.metadata.get("page", None)
        snippet = (d.page_content or "").strip()

        header = f"[{i}] Source: {os.path.basename(src)}"
        if page is not None:
            header += f" | page {page}"

        if len(snippet) > 1800:
            snippet = snippet[:1800] + " …(truncated)"

        blocks.append(header + "\n" + snippet)

    context = "\n\n".join(blocks) if blocks else ""
    return context, used


def _build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def is_low_confidence(best_score: float) -> bool:
    return best_score > LOW_CONF_THRESHOLD


# ------------------------------------------------------------
# Load docs
# ------------------------------------------------------------
def load_docs():
    docs = []
    paths = sorted(glob.glob(DOCUMENT_GLOB))

    if not paths:
        print(f"No files found matching: {DOCUMENT_GLOB}")
        return []

    print("Found PDFs:")
    for p in paths:
        print("  -", p)

    print("\nApplying whitelist filter...\n")

    for path in paths:
        try:
            if not is_whitelisted(path):
                print(f"  ! Not whitelisted (skipping): {path}")
                continue

            if is_encrypted_pdf(path):
                print(f"  ! Encrypted PDF (skipping): {path}")
                continue

            print(f"\nLoading: {path}", flush=True)
            file_docs = load_pdf_with_timeout(path, PDF_LOAD_TIMEOUT_SECONDS)

            for d in file_docs:
                d.metadata["source"] = path

            docs.extend(file_docs)
            print(f"Loaded: {path}  (pages: {len(file_docs)})", flush=True)

        except TimeoutError:
            print(f"  ! Timed out (skipping): {path}", flush=True)
            continue
        except Exception as e:
            print(f"  ! Failed to load {path}: {e}", flush=True)
            continue

    return docs


# ------------------------------------------------------------
# Build / load / save FAISS
# ------------------------------------------------------------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"\nCreated {len(chunks)} chunks for indexing.")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def try_load_vectorstore():
    if not os.path.isdir(INDEX_DIR):
        return None
    try:
        embeddings = OpenAIEmbeddings()
        vs = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception as e:
        print(f"  ! Failed to load existing FAISS index from '{INDEX_DIR}': {e}")
        return None


def save_vectorstore(vectorstore):
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"\nSaved FAISS index to: {INDEX_DIR}/")


# ------------------------------------------------------------
# Answerer (legal-safe RAG)
# ------------------------------------------------------------
def build_answerer(vectorstore):
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_template(
        """You are a real-estate document analysis assistant. Answer the question using ONLY the provided excerpts.

Rules:
- Do NOT provide legal advice. Provide document-grounded analysis only.
- Prefer quoting/paraphrasing clauses precisely.
- If the excerpts do NOT contain enough to answer, say exactly: "I don't know."
- Cite sources inline using [1], [2], etc, matching the excerpt numbers.
- Keep the answer concise, neutral, and specific to the documents.

Excerpts:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()

    def answer_query(query: str, pre_scored: Optional[List[Tuple[Any, float]]] = None) -> Tuple[str, List[Any]]:
        scored = pre_scored if pre_scored is not None else vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
        ranked = _rerank(query, scored)
        context, used = _format_context(ranked, CONTEXT_K)

        if not context.strip():
            return "I don't know.", []

        ans = chain.invoke({"context": context, "question": query}).strip()
        if not ans:
            ans = "I don't know."
        return ans, used

    return answer_query


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of FAISS index")
    args = parser.parse_args()

    print("⚠️  Document analysis only. Not legal advice.\n")

    print(f"Scanning PDFs: {DOCUMENT_GLOB}\n")
    print("Whitelist keywords:", ", ".join(WHITELIST_KEYWORDS))
    print(f"Index dir: {os.path.abspath(INDEX_DIR)}")
    print()

    vectorstore = None

    if not args.rebuild:
        vectorstore = try_load_vectorstore()
        if vectorstore is not None:
            print(f"Loaded existing FAISS index from '{INDEX_DIR}/' ✅\n")

    if vectorstore is None:
        docs = load_docs()
        if not docs:
            print("\nNo loadable PDFs matched the whitelist.")
            return

        print("\nBuilding FAISS index (this will call OpenAI for embeddings)...")
        vectorstore = build_vectorstore(docs)
        save_vectorstore(vectorstore)

    print("\nIndex ready. Semantic search over whitelisted real-estate PDFs.")
    print("Type 'quit' to exit.\n")

    answer_query = build_answerer(vectorstore)

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break

        scored = vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
        if not scored:
            print("No matches found.")
            continue

        best_doc, best_score = scored[0]
        if DEBUG_SCORES:
            print(f"DEBUG best_score={float(best_score):.4f} (threshold={LOW_CONF_THRESHOLD})")

        # Keep this gate lightweight (distance isn't perfect, but prevents total garbage)
        if is_low_confidence(float(best_score)):
            print("Please ask a question related to the indexed real-estate documents.")
            continue

        print("\n🔎 Searching & answering...\n")
        answer, used_docs = answer_query(query, pre_scored=scored)

        print("Answer:\n")
        print(indent(answer.strip(), "  "))

        print("\nSources:\n")
        seen = set()
        for i, item in enumerate(used_docs, start=1):
            d = item["doc"]
            src = d.metadata.get("source", "<unknown>")
            page = d.metadata.get("page")

            key = (src, page)
            if key in seen:
                continue
            seen.add(key)

            if page is not None:
                print(f"  [{i}] {src} (page {page})")
            else:
                print(f"  [{i}] {src}")
        print()

    print("Bye.")


if __name__ == "__main__":
    main()

