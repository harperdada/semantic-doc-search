#!/usr/bin/env python3
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
WHITELIST_KEYWORDS = [
    "cxl",
    "pcie",
    "pci express",
    "kernel",
    "linux",
    "unix",
    "device",
    "driver",
    "devicetree",
    "device-tree",
    "zephyr",
    "optee",
    "u-boot",
    "boot",
    "architecture",
    "programming",
    "operating system",
    "tracing",
    "cuda",
    "pytorch",
    "robotics",
    "datasheet",
    "spec",
    "specification",
]

PDF_LOAD_TIMEOUT_SECONDS = 600

# Where we persist the FAISS index
INDEX_DIR = "faiss_index"

# Retrieval settings
RETRIEVE_K = 12          # pull more, then rerank
CONTEXT_K = 5            # feed best N to the LLM

# Confidence gate for retrieval scores (FAISS distance; lower is better in typical LangChain FAISS)
# Tune by inspecting DEBUG best_score for good vs junk queries.
LOW_CONF_THRESHOLD = float(os.environ.get("LOW_CONF_THRESHOLD", "0.95"))
DEBUG_SCORES = os.environ.get("DEBUG_SCORES", "1").lower() in ("1", "true", "yes")

# Source bias (prefer these in answers if present)
SOURCE_PRIORITY_PATTERNS = [
    "cxl specification",
    "compute express link",
    "cxl overview",
    "pci express",
    "pcie",
]

# Domain signals (used for reranking)
CXL_SIGNALS = [
    "type-3", "type 3", "type3",
    "cxl.mem", "cxl.cache", "cxl.io",
    "hdm", "decoder", "decoders",
    "spa", "system physical address",
    "host-managed", "device memory",
    "coherent", "cache coherent", "coherency",
    "load/store", "load store",
    "dvsec", "capability", "capabilities",
]

OS_SIGNALS = [
    "acpi", "cedt", "srat", "hmat",
    "numa", "hot-plug", "hotplug",
    "memory tier", "tiered",
    "linux", "kernel", "driver",
    "region", "memdev",
]


# ------------------------------------------------------------
# Filters / helpers
# ------------------------------------------------------------
IGNORE_ALWAYS = [
    "settlement",
]


def is_whitelisted(path: str) -> bool:
    name = os.path.basename(path).lower()

    # 1) Hard ignore (always skip)
    if any(k in name for k in IGNORE_ALWAYS):
        return False

    # 2) Whitelist allow
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

    Note: LangChain FAISS typically returns L2 distance (lower is better),
    but we don't hard-depend on polarity for reranking; we mostly sort by our rerank score.
    """
    out: List[Dict[str, Any]] = []
    for doc, score in scored_docs:
        src = doc.metadata.get("source", "")
        txt = doc.page_content or ""

        pri = 6 if _is_priority_source(src) else 0
        cxl = _hits(txt, CXL_SIGNALS)
        osig = _hits(txt, OS_SIGNALS)
        ov = _term_overlap(query, txt)

        # Rerank score: prioritize spec/overview + CXL signals + some overlap
        rer = pri + (1.5 * cxl) + (0.5 * osig) + (0.2 * ov)

        out.append({
            "doc": doc,
            "score": float(score),
            "_rerank": float(rer),
            "_cxl_hits": int(cxl),
            "_os_hits": int(osig),
            "_src_priority": bool(pri),
        })

    # Sort by rerank desc; tie-breaker by smaller absolute distance (heuristic)
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

        # Keep context readable; don't blow up token count
        if len(snippet) > 1800:
            snippet = snippet[:1800] + " …(truncated)"

        blocks.append(header + "\n" + snippet)

    context = "\n\n".join(blocks) if blocks else ""
    return context, used


def _build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def is_low_confidence(best_score: float) -> bool:
    # For typical LangChain FAISS: score is L2 distance => lower is better.
    return best_score > LOW_CONF_THRESHOLD


# ------------------------------------------------------------
# Load docs (PDF only)
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
# Build FAISS vector store
# ------------------------------------------------------------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"\nCreated {len(chunks)} chunks for indexing.")

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def try_load_vectorstore():
    """
    Load FAISS from disk if it exists.
    Note: LangChain requires an embeddings object for query-time embedding.
    """
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
# Build answerer (RAG + synthesis)
# ------------------------------------------------------------
def build_answerer(vectorstore):
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_template(
        """You are a senior technical assistant. Answer the question using ONLY the provided excerpts.
You are allowed (and expected) to synthesize across multiple excerpts even if no single excerpt directly answers the question.

Rules:
- If the excerpts contain enough signals to infer an architectural answer, provide the inferred answer and clearly state any caveats.
- If the excerpts do NOT contain enough to answer, say exactly: "I don't know."
- Cite sources inline using [1], [2], etc, matching the excerpt numbers.
- Keep the answer concise and technical.

Excerpts:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm | StrOutputParser()

    def answer_query(query: str, pre_scored: Optional[List[Tuple[Any, float]]] = None) -> Tuple[str, List[Any]]:
        # Retrieve more, then rerank (reuse pre-scored if provided)
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

    print("\nIndex ready. Semantic search over whitelisted PDFs.")
    print("Type 'quit' to exit.\n")

    answer_query = build_answerer(vectorstore)

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break

        # One retrieval call per query: use it for both confidence + answer.
        scored = vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
        if not scored:
            print("No matches found.")
            continue

        best_doc, best_score = scored[0]
        if DEBUG_SCORES:
            print(f"DEBUG best_score={float(best_score):.4f} (threshold={LOW_CONF_THRESHOLD})")

        if is_low_confidence(float(best_score)):
            print("Please ask a technical question related to the indexed documents.")
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

