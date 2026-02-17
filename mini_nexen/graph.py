from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from . import db
from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    GRAPH_ASSIGN_SIMILARITY_MIN,
    GRAPH_AVG_SIMILARITY_THRESHOLD,
    GRAPH_NOISE_RATIO_THRESHOLD,
    GRAPH_REBUILD_RATIO,
    GRAPH_UNASSIGNED_RATIO_THRESHOLD,
)
from .embeddings import EmbeddingClient, EmbeddingConfig, batch_embed, cosine_similarity, normalize
from .llm import log_task_event
from .text_utils import split_sentences, tokenize

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


@dataclass
class GraphStats:
    total_chunks: int
    assigned_chunks: int
    cluster_count: int
    noise_ratio: float
    avg_similarity: float
    unassigned_ratio: float
    rebuilt: bool


@dataclass
class GraphUpdateResult:
    stats: GraphStats
    new_chunks: int
    pruned_chunks: int
    labels_added: list[str]
    labels_removed: list[str]
    rebuild_attempted: bool
    rebuild_succeeded: bool


def chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    sentences = split_sentences(text or "")
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        sentence_len = len(sentence_tokens)
        if current and current_tokens + sentence_len > max_tokens:
            chunks.append(" ".join(current).strip())
            if overlap > 0:
                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    prev_len = len(tokenize(prev))
                    if overlap_tokens + prev_len > overlap:
                        break
                    overlap_sentences.insert(0, prev)
                    overlap_tokens += prev_len
                current = overlap_sentences
                current_tokens = overlap_tokens
            else:
                current = []
                current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_len
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


class GraphManager:
    def __init__(
        self,
        embed_config: EmbeddingConfig | None,
        llm: object | None = None,
        semantic_labels: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        rebuild_ratio: float = GRAPH_REBUILD_RATIO,
        noise_ratio_threshold: float = GRAPH_NOISE_RATIO_THRESHOLD,
        avg_similarity_threshold: float = GRAPH_AVG_SIMILARITY_THRESHOLD,
        unassigned_ratio_threshold: float = GRAPH_UNASSIGNED_RATIO_THRESHOLD,
        assign_similarity_min: float = GRAPH_ASSIGN_SIMILARITY_MIN,
    ):
        self.embed_config = embed_config
        self.llm = llm
        self.semantic_labels = semantic_labels
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rebuild_ratio = rebuild_ratio
        self.noise_ratio_threshold = noise_ratio_threshold
        self.avg_similarity_threshold = avg_similarity_threshold
        self.unassigned_ratio_threshold = unassigned_ratio_threshold
        self.assign_similarity_min = assign_similarity_min

    def _client(self) -> EmbeddingClient | None:
        if not self.embed_config:
            return None
        return EmbeddingClient(self.embed_config)

    def update_graph(self, docs: list[db.Document]) -> GraphUpdateResult | None:
        db.init_db()
        pruned_chunks = self._prune_archived_chunks()
        new_chunk_count = self._ensure_chunks(docs)
        total_chunks = self._count_chunks()
        if total_chunks == 0:
            return None

        last_metrics = self._load_metrics()
        quality_drop = False
        if last_metrics:
            if last_metrics.get("noise_ratio", 0.0) > self.noise_ratio_threshold:
                quality_drop = True
            if last_metrics.get("avg_similarity", 1.0) < self.avg_similarity_threshold:
                quality_drop = True
            if last_metrics.get("unassigned_ratio", 0.0) > self.unassigned_ratio_threshold:
                quality_drop = True

        clusters_exist = self._count_clusters() > 0
        rebuild = False
        if not clusters_exist and total_chunks >= 5:
            rebuild = True
        elif total_chunks > 0 and (new_chunk_count / total_chunks) >= self.rebuild_ratio:
            rebuild = True
        elif quality_drop:
            rebuild = True
        if pruned_chunks > 0:
            rebuild = True

        labels_before: list[str] = []
        if rebuild:
            labels_before = self._load_cluster_labels()

        rebuild_succeeded = False
        if rebuild:
            log_task_event("Graph: rebuilding clusters")
            rebuild_succeeded = self._rebuild_clusters()
            stats = self._compute_metrics(rebuilt=True)
        else:
            log_task_event("Graph: incremental update")
            self._incremental_assign()
            stats = self._compute_metrics(rebuilt=False)

        if stats:
            self._store_metrics(stats)
        labels_after = self._load_cluster_labels()
        labels_added = sorted(set(labels_after) - set(labels_before))
        labels_removed = sorted(set(labels_before) - set(labels_after))
        if stats:
            return GraphUpdateResult(
                stats=stats,
                new_chunks=new_chunk_count,
                pruned_chunks=pruned_chunks,
                labels_added=labels_added,
                labels_removed=labels_removed,
                rebuild_attempted=rebuild,
                rebuild_succeeded=rebuild_succeeded,
            )
        return None

    def search_documents(self, query: str, top_k: int, top_clusters: int = 3) -> list[db.Document]:
        if not query.strip():
            return []
        clusters = self._load_clusters()
        if not clusters:
            return []
        client = self._client()
        if not client:
            return []
        query_vecs = client.embed_texts([query])
        if len(query_vecs) != 1:
            return []
        query_vec = normalize(query_vecs[0])
        scored_clusters: list[tuple[float, str]] = []
        for cluster_id, centroid in clusters:
            sim = cosine_similarity(query_vec, centroid)
            scored_clusters.append((sim, cluster_id))
        scored_clusters.sort(key=lambda item: item[0], reverse=True)
        top_limit = max(1, int(top_clusters)) if top_clusters else 3
        top_clusters = [cid for sim, cid in scored_clusters[:top_limit] if sim >= 0.2]
        if not top_clusters:
            return []
        doc_scores: dict[str, float] = {}
        with db._connect() as conn:
            for cluster_id in top_clusters:
                rows = conn.execute(
                    """
                    SELECT doc_id, similarity
                    FROM chunks
                    WHERE cluster_id = ?
                    ORDER BY similarity DESC
                    LIMIT 20
                    """,
                    (cluster_id,),
                ).fetchall()
                for row in rows:
                    doc_id = row["doc_id"]
                    score = float(row["similarity"] or 0.0)
                    current = doc_scores.get(doc_id, 0.0)
                    if score > current:
                        doc_scores[doc_id] = score
        ordered = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        doc_ids = [doc_id for doc_id, _ in ordered[:top_k]]
        return db.get_documents_by_ids(doc_ids)

    def score_clusters(self, query: str) -> list[tuple[float, str, str]]:
        if not query.strip():
            return []
        clusters = self._load_clusters_with_ids()
        if not clusters:
            return []
        client = self._client()
        if not client:
            return []
        query_vecs = client.embed_texts([query])
        if len(query_vecs) != 1:
            return []
        query_vec = normalize(query_vecs[0])
        scored: list[tuple[float, str, str]] = []
        for cluster_id, label, centroid in clusters:
            sim = cosine_similarity(query_vec, centroid)
            scored.append((sim, cluster_id, label))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def suggest_interests(self, query: str, limit: int = 3) -> list[str]:
        if not query.strip():
            return []
        clusters = self._load_clusters_with_labels()
        if not clusters:
            return []
        client = self._client()
        if not client:
            return []
        query_vecs = client.embed_texts([query])
        if len(query_vecs) != 1:
            return []
        query_vec = normalize(query_vecs[0])
        scored: list[tuple[float, str]] = []
        for label, centroid in clusters:
            sim = cosine_similarity(query_vec, centroid)
            if sim >= 0.2 and self._is_informative_label(label):
                scored.append((sim, label))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [label for _, label in scored[:limit]]

    def map_topic_to_cluster(self, topic: str) -> tuple[str, str, float] | None:
        if not topic.strip():
            return None
        clusters = self._load_clusters_with_ids()
        if not clusters:
            return None
        client = self._client()
        if not client:
            return None
        query_vecs = client.embed_texts([topic])
        if len(query_vecs) != 1:
            return None
        query_vec = normalize(query_vecs[0])
        best_sim = -1.0
        best_cluster_id = ""
        best_label = ""
        for cluster_id, label, centroid in clusters:
            sim = cosine_similarity(query_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cluster_id
                best_label = label
        if best_sim < 0.2:
            return None
        return best_cluster_id, best_label, best_sim

    def _prune_archived_chunks(self) -> int:
        with db._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM chunks c
                JOIN document_stats s ON c.doc_id = s.doc_id
                WHERE s.archived = 1
                """
            ).fetchone()
            count = int(row["count"] or 0) if row else 0
            if count:
                conn.execute(
                    """
                    DELETE FROM chunks
                    WHERE doc_id IN (SELECT doc_id FROM document_stats WHERE archived = 1)
                    """
                )
        if count:
            log_task_event(f"Graph: pruned {count} chunks from archived documents")
        return count

    def _ensure_chunks(self, docs: list[db.Document]) -> int:
        new_chunks = 0
        now = datetime.now(timezone.utc).isoformat()
        with db._connect() as conn:
            for doc in docs:
                text = db.load_document_text(doc)
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                for idx, chunk in enumerate(chunks):
                    token_count = len(tokenize(chunk))
                    chunk_id = f"{doc.doc_id}:{idx}"
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO chunks
                            (chunk_id, doc_id, chunk_index, text, token_count, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (chunk_id, doc.doc_id, idx, chunk, token_count, now),
                    )
                    if cur.rowcount:
                        new_chunks += 1
        self._embed_missing_chunks()
        return new_chunks

    def _embed_missing_chunks(self) -> None:
        client = self._client()
        if not client:
            return
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE embedding_json IS NULL"
            ).fetchall()
            if not rows:
                return
            texts = [row["text"] for row in rows]
        embeddings = batch_embed(client, texts, batch_size=32)
        if len(embeddings) != len(texts):
            return
        with db._connect() as conn:
            for row, embedding in zip(rows, embeddings):
                conn.execute(
                    "UPDATE chunks SET embedding_json = ? WHERE chunk_id = ?",
                    (json.dumps(embedding), row["chunk_id"]),
                )

    def _rebuild_clusters(self) -> bool:
        client = self._client()
        if not client:
            return False
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text, embedding_json FROM chunks"
            ).fetchall()
        if not rows:
            return False
        embeddings: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for idx, row in enumerate(rows):
            if row["embedding_json"]:
                embeddings.append(json.loads(row["embedding_json"]))
            else:
                embeddings.append(None)
                missing_texts.append(row["text"])
                missing_indices.append(idx)
        if missing_texts:
            extra = batch_embed(client, missing_texts, batch_size=32)
            if len(extra) != len(missing_texts):
                return False
            with db._connect() as conn:
                for offset, emb in zip(missing_indices, extra):
                    embeddings[offset] = emb
                    conn.execute(
                        "UPDATE chunks SET embedding_json = ? WHERE chunk_id = ?",
                        (json.dumps(emb), rows[offset]["chunk_id"]),
                    )
        if any(vec is None for vec in embeddings):
            return False
        embedding_vectors = [vec for vec in embeddings if vec is not None]

        labels = self._cluster_embeddings(embedding_vectors)
        if not labels:
            return False

        normalized = [normalize(vec) for vec in embedding_vectors]
        cluster_map: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            cluster_map.setdefault(label, []).append(idx)
        if not cluster_map:
            log_task_event("Graph: clustering produced no clusters")
            return False

        cluster_ids: dict[int, str] = {}
        cluster_centroids: dict[str, list[float]] = {}
        cluster_sizes: dict[str, int] = {}
        cluster_labels: dict[str, str] = {}
        for label, indices in cluster_map.items():
            cluster_id = str(uuid.uuid4())
            cluster_ids[label] = cluster_id
            centroid = self._mean_vector([normalized[i] for i in indices])
            centroid = normalize(centroid)
            cluster_centroids[cluster_id] = centroid
            cluster_sizes[cluster_id] = len(indices)
            cluster_labels[cluster_id] = self._label_cluster([rows[i]["text"] for i in indices])

        now = datetime.now(timezone.utc).isoformat()
        with db._connect() as conn:
            conn.execute("DELETE FROM clusters")
            for cluster_id, centroid in cluster_centroids.items():
                conn.execute(
                    """
                    INSERT INTO clusters (cluster_id, label, centroid_json, size, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster_id,
                        cluster_labels.get(cluster_id, "cluster"),
                        json.dumps(centroid),
                        cluster_sizes.get(cluster_id, 0),
                        now,
                        now,
                    ),
                )

            for idx, row in enumerate(rows):
                label = labels[idx] if idx < len(labels) else -1
                if label == -1 or label not in cluster_ids:
                    conn.execute(
                        "UPDATE chunks SET cluster_id = NULL, similarity = 0 WHERE chunk_id = ?",
                        (row["chunk_id"],),
                    )
                    continue
                cluster_id = cluster_ids[label]
                sim = cosine_similarity(normalized[idx], cluster_centroids[cluster_id])
                conn.execute(
                    "UPDATE chunks SET cluster_id = ?, similarity = ? WHERE chunk_id = ?",
                    (cluster_id, sim, row["chunk_id"]),
                )
        return True

    def _incremental_assign(self) -> None:
        clusters = self._load_clusters()
        if not clusters:
            return
        client = self._client()
        if not client:
            return
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text, embedding_json FROM chunks WHERE cluster_id IS NULL"
            ).fetchall()
        if not rows:
            return
        embeddings: list[list[float]] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for idx, row in enumerate(rows):
            if row["embedding_json"]:
                embeddings.append(json.loads(row["embedding_json"]))
            else:
                embeddings.append([])
                missing_texts.append(row["text"])
                missing_indices.append(idx)
        if missing_texts:
            extra = batch_embed(client, missing_texts, batch_size=32)
            if len(extra) == len(missing_texts):
                for offset, emb in zip(missing_indices, extra):
                    embeddings[offset] = emb
                with db._connect() as conn:
                    for offset, emb in zip(missing_indices, extra):
                        conn.execute(
                            "UPDATE chunks SET embedding_json = ? WHERE chunk_id = ?",
                            (json.dumps(emb), rows[offset]["chunk_id"]),
                        )

        cluster_ids = [cid for cid, _ in clusters]
        centroids = [centroid for _, centroid in clusters]
        normalized_centroids = [normalize(vec) for vec in centroids]
        current_centroids = {cid: list(vec) for cid, vec in clusters}
        current_sizes: dict[str, int] = {}
        with db._connect() as conn:
            for cid in cluster_ids:
                row = conn.execute(
                    "SELECT size FROM clusters WHERE cluster_id = ?",
                    (cid,),
                ).fetchone()
                current_sizes[cid] = int(row["size"] or 0) if row else 0
        with db._connect() as conn:
            for idx, row in enumerate(rows):
                vec = embeddings[idx]
                if not vec:
                    continue
                vec = normalize(vec)
                best_sim = -1.0
                best_idx = -1
                for c_idx, centroid in enumerate(normalized_centroids):
                    sim = cosine_similarity(vec, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = c_idx
                if best_idx == -1 or best_sim < self.assign_similarity_min:
                    conn.execute(
                        "UPDATE chunks SET cluster_id = NULL, similarity = 0 WHERE chunk_id = ?",
                        (row["chunk_id"],),
                    )
                    continue
                cluster_id = cluster_ids[best_idx]
                conn.execute(
                    "UPDATE chunks SET cluster_id = ?, similarity = ? WHERE chunk_id = ?",
                    (cluster_id, best_sim, row["chunk_id"]),
                )
                base_size = current_sizes.get(cluster_id, 0)
                new_size = base_size + 1
                current_centroids[cluster_id] = self._merge_centroid(
                    current_centroids[cluster_id], vec, base_size
                )
                current_sizes[cluster_id] = new_size

            now = datetime.now(timezone.utc).isoformat()
            for cluster_id, centroid in current_centroids.items():
                size = current_sizes.get(cluster_id, 0)
                conn.execute(
                    """
                    UPDATE clusters
                    SET centroid_json = ?, size = ?, updated_at = ?
                    WHERE cluster_id = ?
                    """,
                    (json.dumps(normalize(centroid)), size, now, cluster_id),
                )
        self._refresh_cluster_labels([cid for cid, _ in clusters])

    def _cluster_embeddings(self, embeddings: list[list[float]]) -> list[int]:
        if not embeddings:
            return []
        try:
            import numpy as np  # type: ignore
        except Exception:
            log_task_event("Graph: numpy unavailable; using single-cluster fallback.")
            return [0] * len(embeddings)
        vectors = np.array([normalize(vec) for vec in embeddings], dtype=float)
        try:
            import hdbscan  # type: ignore
            min_size = max(5, int(math.sqrt(len(embeddings))))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric="euclidean")
            labels = [int(x) for x in clusterer.fit_predict(vectors)]
            unique = {label for label in labels if label != -1}
            if unique:
                return labels
            log_task_event("Graph: HDBSCAN produced no clusters; falling back to KMeans.")
        except Exception:
            log_task_event("Graph: HDBSCAN unavailable; falling back to KMeans.")
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except Exception:
            log_task_event("Graph: scikit-learn unavailable; using single-cluster fallback.")
            return [0] * len(embeddings)
        k = max(2, int(math.sqrt(len(embeddings) / 2)))
        k = min(k, max(2, len(embeddings) // 2))
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(vectors)
        return [int(x) for x in labels]

    def _mean_vector(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        length = len(vectors[0])
        acc = [0.0] * length
        for vec in vectors:
            for idx, value in enumerate(vec):
                acc[idx] += value
        return [value / len(vectors) for value in acc]

    def _merge_centroid(self, centroid: list[float], vec: list[float], base_size: int) -> list[float]:
        if not centroid:
            return vec
        if base_size < 0:
            base_size = 0
        total = base_size + 1
        return [
            (centroid[i] * base_size + vec[i]) / total for i in range(len(centroid))
        ]

    def _label_cluster(self, texts: list[str]) -> str:
        if not texts:
            raise RuntimeError("LLM-based cluster labels require non-empty cluster text.")
        if not (self.semantic_labels and self.llm):
            raise RuntimeError("LLM-based cluster labels are required but not configured.")
        label = self._label_cluster_semantic(texts)
        if not label:
            raise RuntimeError("LLM returned an empty cluster label.")
        return label

    def _label_cluster_semantic(self, texts: list[str]) -> str:
        if not self.llm:
            return ""
        samples = []
        for text in texts[:5]:
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            if snippet:
                samples.append(snippet)
        if not samples:
            return ""
        prompt = (
            "Label this topic cluster in 2-6 words. "
            "Return only the label, no punctuation, no quotes."
            "\n\n"
            "Cluster samples:\n- " + "\n- ".join(samples)
        )
        response = self.llm.generate(
            system_prompt="You name topic clusters.",
            user_prompt=prompt,
            task="cluster label",
            agent="Clusterer",
        )
        label = (response or "").strip().strip("\"'`")
        label = label.replace("\n", " ").strip()
        if len(label) > 80:
            label = label[:80].strip()
        return label

    def _refresh_cluster_labels(self, cluster_ids: list[str]) -> None:
        if not self.semantic_labels or not self.llm or not cluster_ids:
            return
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT cluster_id, label FROM clusters WHERE cluster_id IN ({})".format(
                    ",".join("?" for _ in cluster_ids)
                ),
                tuple(cluster_ids),
            ).fetchall()
        for row in rows:
            cluster_id = row["cluster_id"]
            label = row["label"] or ""
            if self._is_informative_label(label):
                continue
            with db._connect() as conn:
                texts = conn.execute(
                    "SELECT text FROM chunks WHERE cluster_id = ? LIMIT 5",
                    (cluster_id,),
                ).fetchall()
            samples = [text_row["text"] for text_row in texts if text_row["text"]]
            if not samples:
                continue
            new_label = self._label_cluster(samples)
            if new_label and new_label != label:
                with db._connect() as conn:
                    conn.execute(
                        "UPDATE clusters SET label = ? WHERE cluster_id = ?",
                        (new_label, cluster_id),
                    )
                log_task_event(f"Graph: relabeled cluster {cluster_id[:8]} -> '{new_label}'")

    def _is_informative_token(self, token: str) -> bool:
        if len(token) < 3:
            return False
        if token in _STOPWORDS:
            return False
        has_alpha = any(ch.isalpha() for ch in token)
        return has_alpha

    def _is_informative_label(self, label: str) -> bool:
        if not label:
            return False
        if label.strip().lower() in {"cluster", "misc", "unknown"}:
            return False
        parts = [part.strip().lower() for part in label.split("/") if part.strip()]
        if not parts:
            return False
        return any(self._is_informative_token(part) for part in parts)

    def _compute_metrics(self, rebuilt: bool) -> GraphStats | None:
        total = self._count_chunks()
        if total == 0:
            return None
        assigned = self._count_assigned_chunks()
        unassigned = total - assigned
        avg_similarity = self._avg_similarity()
        noise_ratio = unassigned / total if total else 0.0
        unassigned_ratio = noise_ratio
        cluster_count = self._count_clusters()
        return GraphStats(
            total_chunks=total,
            assigned_chunks=assigned,
            cluster_count=cluster_count,
            noise_ratio=noise_ratio,
            avg_similarity=avg_similarity,
            unassigned_ratio=unassigned_ratio,
            rebuilt=rebuilt,
        )

    def _store_metrics(self, stats: GraphStats) -> None:
        payload = {
            "total_chunks": stats.total_chunks,
            "assigned_chunks": stats.assigned_chunks,
            "cluster_count": stats.cluster_count,
            "noise_ratio": stats.noise_ratio,
            "avg_similarity": stats.avg_similarity,
            "unassigned_ratio": stats.unassigned_ratio,
            "rebuilt": stats.rebuilt,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with db._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?, ?)",
                ("metrics", json.dumps(payload)),
            )

    def _load_metrics(self) -> dict[str, float] | None:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT value FROM graph_meta WHERE key = ?",
                ("metrics",),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["value"])
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _count_chunks(self) -> int:
        with db._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"] or 0)

    def _count_assigned_chunks(self) -> int:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE cluster_id IS NOT NULL"
            ).fetchone()
        return int(row["count"] or 0)

    def _avg_similarity(self) -> float:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT AVG(similarity) AS avg_sim FROM chunks WHERE cluster_id IS NOT NULL"
            ).fetchone()
        return float(row["avg_sim"] or 0.0)

    def _count_clusters(self) -> int:
        with db._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM clusters").fetchone()
        return int(row["count"] or 0)

    def _load_clusters(self) -> list[tuple[str, list[float]]]:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT cluster_id, centroid_json FROM clusters"
            ).fetchall()
        clusters = []
        for row in rows:
            try:
                centroid = json.loads(row["centroid_json"])
            except Exception:
                centroid = []
            if centroid:
                clusters.append((row["cluster_id"], centroid))
        return clusters

    def _load_cluster_labels(self) -> list[str]:
        with db._connect() as conn:
            rows = conn.execute("SELECT label FROM clusters").fetchall()
        labels = []
        for row in rows:
            label = row["label"]
            if label:
                labels.append(label)
        return labels

    def _load_clusters_with_labels(self) -> list[tuple[str, list[float]]]:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT label, centroid_json FROM clusters"
            ).fetchall()
        clusters = []
        for row in rows:
            try:
                centroid = json.loads(row["centroid_json"])
            except Exception:
                centroid = []
            label = row["label"] or ""
            if centroid and label:
                clusters.append((label, centroid))
        return clusters

    def _load_clusters_with_ids(self) -> list[tuple[str, str, list[float]]]:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT cluster_id, label, centroid_json FROM clusters"
            ).fetchall()
        clusters = []
        for row in rows:
            try:
                centroid = json.loads(row["centroid_json"])
            except Exception:
                centroid = []
            label = row["label"] or ""
            if centroid and label:
                clusters.append((row["cluster_id"], label, centroid))
        return clusters
