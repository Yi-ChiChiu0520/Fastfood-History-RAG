# retriever.py
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# For type hints only; you create/pass this from your app
from chromadb.api.models import Collection

class DocRetriever:
    """
    Document-level retriever that:
    1) queries many chunks from Chroma,
    2) reranks with a cross-encoder,
    3) aggregates scores by doc_id,
    4) returns top documents with supporting snippets.
    """

    def __init__(
        self,
        collection: Collection,
        reranker_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
    ):
        self.collection = collection

        # --- Load reranker once (faster than loading every query) ---
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.reranker_model.to(self.device)
        self.reranker_model.eval()

    # ---------- Internal helpers ----------

    @torch.no_grad()
    def _cross_scores(self, query: str, passages: List[str]) -> List[float]:
        """
        Compute cross-encoder scores for (query, passage) pairs.
        Higher is better.
        """
        # Batch for speed (optional simple batching)
        scores: List[float] = []
        batch_size = 8
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            inputs = self.reranker_tokenizer(
                [query] * len(batch),
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.reranker_model(**inputs).logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())
        return scores

    def _aggregate_by_doc(
        self,
        scored_items: List[Tuple[str, float, str, Dict[str, Any]]],
        top_docs: int,
        evidences_per_doc: int,
        top_m_for_score: int,
    ):
        """
        Group (doc_id, score, text, metadata) by doc, compute doc_score, and return
        top docs with their best evidences.
        """
        grouped: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = defaultdict(list)
        for doc_id, score, text, md in scored_items:
            grouped[doc_id].append((score, text, md))

        def doc_score(items: List[Tuple[float, str, Dict[str, Any]]]) -> float:
            items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
            top_scores = [s for s, _, _ in items_sorted[:top_m_for_score]]
            votes = len(items_sorted)
            return sum(top_scores) + 0.1 * votes  # small vote bonus

        ranked = sorted(grouped.items(), key=lambda kv: doc_score(kv[1]), reverse=True)[:top_docs]

        results = []
        for doc_id, items in ranked:
            items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
            evidences = [
                {
                    "score": round(s, 4),
                    "snippet": t[:500],
                    "metadata": m,
                }
                for s, t, m in items_sorted[:evidences_per_doc]
            ]
            results.append(
                {
                    "doc_id": doc_id,
                    "doc_score": round(doc_score(items), 4),
                    "evidences": evidences,
                }
            )
        return results

    # ---------- Public API ----------

    def retrieve_docs(
        self,
        query: str,
        k_chunks: int = 50,
        top_docs: int = 5,
        evidences_per_doc: int = 3,
        top_m_for_score: int = 3,
        include_sources: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        1) Query many chunks from Chroma
        2) Cross-encode rerank
        3) Aggregate by doc_id and return top documents + evidences
        """
        res = self.collection.query(
            query_texts=[query],
            n_results=k_chunks,
            include=["documents", "metadatas"] + (["distances"] if not include_sources else []),
        )

        docs_list: List[str] = res["documents"][0]
        metas_list: List[Dict[str, Any]] = res["metadatas"][0]

        # Ensure doc_id exists (expect it was added during upsert)
        for md in metas_list:
            if "doc_id" not in md:
                # fallback: derive from source basename
                src = md.get("source", "")
                md["doc_id"] = os.path.splitext(os.path.basename(src))[0] or src

        # Cross-encoder rerank on returned chunks
        scores = self._cross_scores(query, docs_list)

        scored_items: List[Tuple[str, float, str, Dict[str, Any]]] = []
        for text, md, score in zip(docs_list, metas_list, scores):
            scored_items.append((md["doc_id"], float(score), text, md))

        # Aggregate at the document level
        return self._aggregate_by_doc(
            scored_items,
            top_docs=top_docs,
            evidences_per_doc=evidences_per_doc,
            top_m_for_score=top_m_for_score,
        )

    @staticmethod
    def build_context(results: List[Dict[str, Any]], max_chars: int = 2500) -> str:
        """
        Build a compact context string from the doc-level evidences.
        """
        parts: List[str] = []
        for r in results:
            parts.append(f"[DOC] {r['doc_id']} (score={r['doc_score']})")
            for ev in r["evidences"]:
                src = ev["metadata"].get("source", "")
                page = ev["metadata"].get("page", None)
                loc = f"{src}#page={page}" if page is not None else src
                parts.append(f" - ev={ev['score']} | {loc}\n   {ev['snippet']}")
            parts.append("")  # blank line

        context = "\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"
        return context
