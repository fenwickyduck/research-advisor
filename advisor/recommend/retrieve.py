"""Candidate retrieval: from what you have read to what you should read next.

Four steps, all local and all cheap:

1. **Preference vectors.** Rocchio-style — pull toward what you liked, push away
   from what you did not. Rather than one average, k-means over your liked
   papers gives several *interest clusters*, because a single centroid over
   someone working on both lattices and MPC lands between the two and retrieves
   neither well.
2. **Cosine search** against every embedded paper, with a mild recency prior.
3. **MMR** to drop near-duplicates, so a batch is not eight variations on one
   result.
4. **Followed authors**, retrieved by name and merged in ahead of the rest —
   the one part of a batch that does not go through the vector space at all.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from advisor.config import Config
from advisor.embed import store
from advisor.recommend import feedback

RECENT_WINDOW_DAYS = 548  # ~18 months
# How many library papers one explicit thumbs-up is worth when shaping interests.
LIKED_WEIGHT = 3.0
# Longest source title quoted back in an attribution before it is elided.
TITLE_CHARS = 90
# Score given to a followed author's paper. Above the cosine range on purpose:
# it is not a similarity, and it must not be sorted against one.
FOLLOWED_SCORE = 1.5


@dataclass
class Candidate:
    paper_id: int
    score: float
    # Set when the paper was retrieved by author rather than by similarity, so
    # the feed can say why truthfully instead of quoting a cosine that had no
    # part in choosing it.
    via: str | None = None


@dataclass
class Attribution:
    """A paper of yours that a recommendation was matched to."""

    paper_id: int
    title: str
    similarity: float
    kind: str  # "read" if it is in your library, "liked" if you rated it up

    def sentence(self) -> str:
        """The attribution as it is stored and shown."""
        title = self.title
        if len(title) > TITLE_CHARS:
            title = title[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"
        return (
            f"Closest to “{title}”, which you "
            f"{'read' if self.kind == 'read' else 'liked'} (cosine {self.similarity:.2f})."
        )


# --------------------------------------------------------------- your preferences


def rated_ids(conn: sqlite3.Connection) -> tuple[list[int], list[int]]:
    """Papers you liked and disliked, using each paper's most recent rating.

    Feedback is append-only so you can change your mind; only the latest verdict
    on any paper counts.
    """
    rows = conn.execute(
        """SELECT paper_id, rating FROM feedback f
            WHERE f.id = (SELECT max(id) FROM feedback g WHERE g.paper_id = f.paper_id)"""
    ).fetchall()

    liked = [row["paper_id"] for row in rows if row["rating"] > 0]
    disliked = [row["paper_id"] for row in rows if row["rating"] < 0]
    return liked, disliked


def library_ids(conn: sqlite3.Connection, statuses: tuple[str, ...] = ("read",)) -> list[int]:
    placeholders = ",".join("?" * len(statuses))
    return [
        row["paper_id"]
        for row in conn.execute(
            f"SELECT paper_id FROM library WHERE status IN ({placeholders})", statuses
        )
    ]


def _vectors_for(
    conn: sqlite3.Connection, matrix: np.ndarray, paper_ids: list[int]
) -> np.ndarray:
    rows = store.rows_for(conn, paper_ids)
    indices = [rows[pid] for pid in paper_ids if pid in rows and rows[pid] < matrix.shape[0]]
    if not indices:
        return np.empty((0, matrix.shape[1]), dtype=store.DTYPE)
    return np.asarray(matrix[indices], dtype=store.DTYPE)


def interest_clusters(
    positives: np.ndarray, k: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """Cluster the papers you like into a few centroids.

    One averaged taste vector is a poor summary of someone who reads across
    several subfields; k centroids keep those interests distinct so each gets
    its own share of the results. ``weights`` lets an explicitly-rated paper
    pull harder than one that is merely in your library.
    """
    if positives.shape[0] == 0:
        return positives
    if positives.shape[0] <= k:
        return store.normalize(positives)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(positives, sample_weight=weights)
    return store.normalize(kmeans.cluster_centers_)


def _profile_steer(
    conn: sqlite3.Connection, cfg: Config, dims: int
) -> tuple[np.ndarray, np.ndarray]:
    """The profile's steering vectors, or nothing if they cannot be used.

    Never allowed to break a run: an encoder that is not installed, or vectors
    left over from a different embedding model, mean the profile is ignored for
    this run rather than that no recommendations are produced at all.
    """
    from advisor.recommend import profile

    empty = np.empty((0, dims), dtype=store.DTYPE)
    try:
        wanted, unwanted = profile.steer_vectors(conn, cfg)
    except Exception:
        return empty, empty

    if wanted.shape[0] and wanted.shape[1] != dims:
        return empty, empty
    if unwanted.shape[0] and unwanted.shape[1] != dims:
        return empty, empty
    return wanted, unwanted


def preference_vectors(
    conn: sqlite3.Connection, matrix: np.ndarray, cfg: Config
) -> np.ndarray:
    """Build the query vectors to search with.

    Your library and your ratings are combined rather than one replacing the
    other. Everything you have read is a standing signal; an explicit thumbs-up
    is a stronger one on top of it, worth ``LIKED_WEIGHT`` library papers.
    Treating a rating as a *replacement* meant a single click could blank out a
    whole reading history — the first thumbs-up would leave the advisor
    recommending from that one paper alone.

    Anything you have explicitly disliked is dropped from the positive side even
    if it sits in your library, since having read something is not endorsement.
    """
    liked, disliked = rated_ids(conn)
    disliked_set = set(disliked)

    in_library = [pid for pid in library_ids(conn) if pid not in disliked_set]
    liked_only = [pid for pid in liked if pid not in in_library]
    paper_ids = in_library + liked_only

    positives = _vectors_for(conn, matrix, paper_ids)

    liked_set = set(liked)
    weights = np.array(
        [LIKED_WEIGHT if pid in liked_set else 1.0 for pid in paper_ids],
        dtype=np.float64,
    )[: positives.shape[0]]

    # What the profile asks for, in your own words. Added before clustering so a
    # stated interest can claim a centroid of its own rather than being averaged
    # into the reading it is meant to redirect.
    wanted, unwanted = _profile_steer(conn, cfg, matrix.shape[1])
    if wanted.shape[0]:
        positives = np.vstack([positives, wanted]) if positives.shape[0] else wanted
        weights = np.concatenate(
            [weights, np.full(wanted.shape[0], cfg.profile_weight, dtype=np.float64)]
        )

    if positives.shape[0] == 0:
        return np.empty((0, matrix.shape[1]), dtype=store.DTYPE)

    centroids = interest_clusters(positives, cfg.n_clusters, weights)

    # Rocchio: push away from what you rejected — but weighted by *why*. A paper
    # dismissed as already-known was a correct retrieval, so it contributes
    # nothing here; one dismissed as the wrong subfield contributes fully.
    reasons = feedback.latest_for(conn)
    weighted = [
        (pid, feedback.negative_weight(reasons.get(pid, (0, []))[1])) for pid in disliked
    ]
    against_ids = [pid for pid, weight in weighted if weight > 0]

    if against_ids or unwanted.shape[0]:
        negatives = _vectors_for(conn, matrix, against_ids)
        strength_list = [weight for _, weight in weighted if weight > 0][
            : negatives.shape[0]
        ]

        # "Less of" is as deliberate a rejection as there is: you did not merely
        # decline one paper, you named the thing itself. It pushes at full weight.
        if unwanted.shape[0]:
            negatives = (
                np.vstack([negatives, unwanted]) if negatives.shape[0] else unwanted
            )
            strength_list += [1.0] * unwanted.shape[0]

        if negatives.shape[0]:
            strengths = np.array(strength_list, dtype=store.DTYPE)

            # Direction: which way to move, from the weighted mix of rejections.
            against = store.normalize(
                (negatives * strengths[:, None]).sum(axis=0) / max(strengths.sum(), 1e-9)
            )
            # Distance: how far to move. This has to scale beta separately —
            # inside the weighted mean the weight cancels out (w*v / w == v), so
            # a lone "too theoretical" would otherwise push exactly as hard as a
            # lone "wrong subfield" and the reasons would be decorative.
            strength = float(strengths.mean())
            centroids = centroids * cfg.rocchio_alpha - against * cfg.rocchio_beta * strength

    return store.normalize(centroids)


# ------------------------------------------------------------------- the search


def effective_recency_boost(conn: sqlite3.Connection, cfg: Config) -> float:
    """The recency prior, leaned on by "want newer work" feedback.

    Age is the one complaint the vector space genuinely cannot express — an old
    paper is not *semantically* different from a new one on the same topic. So
    that reason is handled here instead, scaling the prior toward newer results
    rather than pushing the query somewhere else.

    Capped at 3x so a run of impatient clicks cannot turn the feed into a pure
    reverse-chronological list and bury a foundational paper you have not read.
    """
    reasons = feedback.latest_for(conn)
    negative = [tags for rating, tags in reasons.values() if rating < 0]
    if not negative:
        return cfg.recency_boost

    wanting_newer = sum(1 for tags in negative if feedback.wants_newer(tags))
    if not wanting_newer:
        return cfg.recency_boost

    scale = 1.0 + 2.0 * (wanting_newer / len(negative))
    return cfg.recency_boost * scale


def _recency_boost(
    conn: sqlite3.Connection, ids: np.ndarray, cfg: Config, boost: float | None = None
) -> np.ndarray:
    """A small bonus for recent papers.

    Deliberately additive and small rather than a filter: a foundational paper
    you never read is exactly what an advisor should surface, so age must not
    disqualify anything outright.
    """
    boost = cfg.recency_boost if boost is None else boost
    if boost <= 0 or len(ids) == 0:
        return np.zeros(len(ids), dtype=store.DTYPE)

    cutoff = (date.today() - timedelta(days=RECENT_WINDOW_DAYS)).isoformat()
    placeholders = ",".join("?" * len(ids))
    recent = {
        row["id"]
        for row in conn.execute(
            f"SELECT id FROM papers WHERE id IN ({placeholders}) AND published_at >= ?",
            [*ids.tolist(), cutoff],
        )
    }
    return np.array(
        [boost if pid in recent else 0.0 for pid in ids.tolist()],
        dtype=store.DTYPE,
    )


def excluded_ids(conn: sqlite3.Connection) -> set[int]:
    """Papers not to recommend: already in your library, or already suggested.

    Recommending something twice is the fastest way to make the feed feel
    broken, so previously shown papers are excluded even if never acted on.
    """
    rows = conn.execute(
        """SELECT paper_id FROM library
           UNION SELECT paper_id FROM recommendations
           UNION SELECT paper_id FROM feedback"""
    )
    return {row["paper_id"] for row in rows}


def search(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    queries: np.ndarray,
    cfg: Config,
    exclude: set[int] | None = None,
) -> list[Candidate]:
    """Score the corpus against each query vector and merge the results."""
    if queries.shape[0] == 0 or matrix.shape[0] == 0:
        return []

    exclude = exclude if exclude is not None else excluded_ids(conn)
    ids_by_row = store.ids_by_row(conn)
    usable = min(matrix.shape[0], len(ids_by_row))

    # (n_queries, n_papers) — both sides are unit vectors, so this is cosine.
    scores = np.asarray(queries, dtype=store.DTYPE) @ np.asarray(
        matrix[:usable], dtype=store.DTYPE
    ).T

    per_query = min(cfg.n_retrieve_per_cluster, usable)
    boost = effective_recency_boost(conn, cfg)
    best: dict[int, float] = {}
    per_cluster: list[list[tuple[int, float]]] = []

    for query_scores in scores:
        # argpartition beats a full sort: we only need the top slice.
        top = np.argpartition(-query_scores, per_query - 1)[:per_query]
        ids = ids_by_row[top]
        boosted = query_scores[top] + _recency_boost(conn, ids, cfg, boost)

        ranked_here: list[tuple[int, float]] = []
        for paper_id, score in sorted(
            zip(ids.tolist(), boosted.tolist()), key=lambda item: item[1], reverse=True
        ):
            if paper_id < 0 or paper_id in exclude:
                continue
            ranked_here.append((paper_id, score))
            # A paper matching several of your interests keeps its best score.
            if score > best.get(paper_id, -2.0):
                best[paper_id] = score
        per_cluster.append(ranked_here)

    # Interleave the clusters rather than sorting everything together.
    #
    # Cosine scores are not comparable across clusters: a dense, well-populated
    # interest yields systematically higher similarities than a sparse one, so a
    # single global sort lets one interest take every slot — which defeats the
    # point of having clustered at all. Round-robin gives each interest its turn.
    merged: list[Candidate] = []
    seen: set[int] = set()
    for rank in range(per_query):
        for ranked_here in per_cluster:
            if rank >= len(ranked_here):
                continue
            paper_id, _ = ranked_here[rank]
            if paper_id in seen:
                continue
            seen.add(paper_id)
            merged.append(Candidate(paper_id, best[paper_id]))

    return merged


def diversify(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    candidates: list[Candidate],
    limit: int,
    lambda_: float = 0.7,
) -> list[Candidate]:
    """Maximal Marginal Relevance: trade a little relevance for variety.

    Without this a shortlist is often eight near-identical papers from one
    group, which wastes both the slots and the ranking call that follows.
    """
    if len(candidates) <= limit:
        return candidates

    rows = store.rows_for(conn, [c.paper_id for c in candidates])
    pool = [c for c in candidates if rows.get(c.paper_id, -1) < matrix.shape[0]]
    if not pool:
        return candidates[:limit]

    vectors = np.asarray(matrix[[rows[c.paper_id] for c in pool]], dtype=store.DTYPE)
    relevance = np.array([c.score for c in pool], dtype=store.DTYPE)

    selected: list[int] = [int(np.argmax(relevance))]
    # Similarity of every candidate to the most similar already-selected one.
    max_sim = vectors @ vectors[selected[0]]

    while len(selected) < limit and len(selected) < len(pool):
        mmr = lambda_ * relevance - (1 - lambda_) * max_sim
        mmr[selected] = -np.inf

        choice = int(np.argmax(mmr))
        selected.append(choice)
        max_sim = np.maximum(max_sim, vectors @ vectors[choice])

    return [pool[i] for i in selected]


def explain(
    conn: sqlite3.Connection, candidates: list[Candidate], cfg: Config
) -> dict[int, Attribution]:
    """Which paper of yours pulled each recommendation in.

    Without a language model there is still an honest answer to "why am I being
    shown this": the nearest paper among the ones you have read or liked. That
    is literally what drove the match, it costs one more matrix product, and it
    is often more checkable than a written justification — you recognise the
    paper it named.

    Returns candidate paper_id -> the source it came from.
    """
    matrix = store.load(cfg.vectors_path)
    if matrix is None or not candidates:
        return {}

    liked, disliked = rated_ids(conn)
    disliked_set = set(disliked)
    in_library = [pid for pid in library_ids(conn) if pid not in disliked_set]
    library_set = set(in_library)
    sources = in_library + [pid for pid in liked if pid not in library_set]
    if not sources:
        return {}

    source_rows = store.rows_for(conn, sources)
    usable = [pid for pid in sources if source_rows.get(pid, matrix.shape[0]) < matrix.shape[0]]
    if not usable:
        return {}

    candidate_rows = store.rows_for(conn, [c.paper_id for c in candidates])
    matched = [
        candidate
        for candidate in candidates
        if candidate_rows.get(candidate.paper_id, matrix.shape[0]) < matrix.shape[0]
    ]
    if not matched:
        return {}

    source_vectors = np.asarray(
        matrix[[source_rows[pid] for pid in usable]], dtype=store.DTYPE
    )
    candidate_vectors = np.asarray(
        matrix[[candidate_rows[c.paper_id] for c in matched]], dtype=store.DTYPE
    )

    # Every (source, candidate) pair in one product; column argmax picks the
    # source that each candidate is nearest to.
    similarities = source_vectors @ candidate_vectors.T
    best = np.argmax(similarities, axis=0)

    titles = {
        row["id"]: row["title"]
        for row in conn.execute(
            f"SELECT id, title FROM papers WHERE id IN ({','.join('?' * len(usable))})",
            usable,
        )
    }

    out: dict[int, Attribution] = {}
    for column, candidate in enumerate(matched):
        row = int(best[column])
        source_id = usable[row]
        out[candidate.paper_id] = Attribution(
            paper_id=source_id,
            title=titles.get(source_id, ""),
            similarity=float(similarities[row, column]),
            kind="read" if source_id in library_set else "liked",
        )

    return out


def followed(conn: sqlite3.Connection, cfg: Config, exclude: set[int]) -> list[Candidate]:
    """New work by people you follow, retrieved by name rather than by vector.

    Deliberately outside the similarity search. A followed author's paper is
    worth seeing *because of who wrote it*, so requiring it to also look like
    your existing library would filter out exactly the case this feature
    exists for — the one that took them somewhere new.
    """
    from advisor import authors

    matches = authors.papers_by(conn, exclude=exclude, limit=cfg.n_followed)
    # Scored above the similarity range so they survive the shortlist cut.
    return [Candidate(pid, FOLLOWED_SCORE, via=who) for pid, who in matches]


def recommend(
    conn: sqlite3.Connection, cfg: Config, limit: int | None = None
) -> list[Candidate]:
    """Full local pipeline: preferences -> search -> diversify, plus follows."""
    matrix = store.load(cfg.vectors_path)
    if matrix is None:
        return []

    exclude = excluded_ids(conn)
    by_author = followed(conn, cfg, exclude)

    queries = preference_vectors(conn, matrix, cfg)
    if queries.shape[0] == 0:
        # A follow list is a complete reason to recommend on its own — no
        # library, no profile, no embeddings needed on the query side.
        return by_author[: limit or cfg.n_candidates]

    candidates = search(conn, matrix, queries, cfg, exclude=exclude)
    similar = diversify(conn, matrix, candidates, limit or cfg.n_candidates)

    # Followed authors first, then similarity, with no paper appearing twice —
    # and the whole thing cut to the limit. Returning follows *on top of* a
    # full similarity batch would silently hand back more than was asked for.
    wanted = limit or cfg.n_candidates
    seen = {c.paper_id for c in by_author}
    merged = by_author + [c for c in similar if c.paper_id not in seen]
    return merged[:wanted]
