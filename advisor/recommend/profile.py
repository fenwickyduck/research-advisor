"""The advisor's model of you: a short markdown document, versioned.

This is the piece that makes the tool an advisor rather than a similarity
search. Embeddings capture "more like this"; only prose can carry "I already
know FHE bootstrapping, stop showing me introductions" or "I'm pivoting toward
MPC in the autumn". Those are exactly the things your free-text notes say, and
exactly what a vector cannot represent.

Two of its sections are not prose but instructions, and retrieval reads them
directly:

    ## More of
    doubly-efficient private information retrieval

    ## Less of
    post-quantum migration policy surveys

Each line is embedded with the same local model that encodes the corpus —
SPECTER takes arbitrary text, not only papers — so a stated interest becomes a
query vector and a stated dislike becomes something to steer away from. That
matters because it needs no API key: writing the profile by hand is enough to
change what you are shown. The remaining sections stay prose, for you
to read and to hand to an assistant when you want one rewritten.

You write it — by hand, or by pasting :func:`briefing` into whatever assistant
you already have and pasting the answer back. Nothing here calls a model or
holds a credential. Every version is kept, so you can see how the picture of
you changed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from advisor.config import Config
from advisor.embed import store
from advisor.models import now

MAX_LIBRARY_SAMPLE = 60
MAX_NOTES = 40

# Headings whose contents steer retrieval, rather than merely describing you.
STEER_SECTIONS = {"more of": "more", "less of": "less"}

# What you get when writing a profile by hand. Sections are left empty on
# purpose: anything sitting under a steering heading is read as an instruction,
# so a worked example in the box would quietly become part of your interests.
TEMPLATE = """\
## Working on


## Background


## More of


## Less of
"""
# A cap, so a runaway profile cannot turn one recommendation into a hundred
# encoder calls. Well above what anyone writes by hand.
MAX_STEER_ITEMS = 12

BRIEF = """\
You maintain a short profile of a researcher, used to decide which papers to \
recommend to them next.

Write the profile as markdown with these sections:

## Working on
What they appear to be actively researching. Be specific — name techniques, \
primitives and subfields, not broad areas.

## Background
What they can be assumed to already know, so introductory or survey material on \
these topics is not worth their time.

## Not interested in
Topics and styles they have rejected, and why, where the evidence supports it.

## Preferences
How they like their reading: theory versus implementation, attacks versus \
constructions, and anything else their notes reveal.

## More of
## Less of
Short topic phrases, one per line, no explanation and no full sentences — \
"doubly-efficient private information retrieval", not "they would like to see \
more work on PIR". These two sections are fed to a text encoder and used \
directly as search directions, so each line should read like the title of a \
paper they do or do not want. Five or so lines each is plenty; leave a section \
empty rather than padding it.

Rules:
- Ground every claim in the evidence given. Do not invent interests.
- Their free-text notes are the strongest signal; weight them above titles.
- Prefer specifics ("lattice-based signatures, NTT implementation") over \
generalities ("cryptography").
- If the evidence is thin, say so plainly rather than padding.
- Keep the whole profile under 400 words. It is a working note, not an essay.
"""


@dataclass
class Profile:
    id: int
    content: str
    written_by: str
    created_at: str


@dataclass
class Steer:
    """The machine-readable half of a profile: what to seek and what to avoid."""

    more: list[str] = field(default_factory=list)
    less: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.more or self.less)

    def key(self, model: str) -> str:
        """Cache identity. Includes the model, since vectors do not survive a change."""
        payload = "\n".join([model, "+", *self.more, "-", *self.less])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse(content: str | None) -> Steer:
    """Pull the steering sections out of a profile, ignoring the prose.

    Deliberately lenient, because this is a document people type into: headings
    match at any level and any case, items may be bullets or bare lines. An
    unrecognised heading switches steering off rather than capturing what
    follows, so prose sections can never be mistaken for instructions — which
    also means a profile written before this existed simply steers nothing.
    """
    found: dict[str, list[str]] = {"more": [], "less": []}
    active: str | None = None

    for raw in (content or "").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            active = STEER_SECTIONS.get(line.lstrip("#").strip().rstrip(":").lower())
            continue
        if active is None or not line:
            continue

        item = line.lstrip("-*•").strip()
        if item and len(found[active]) < MAX_STEER_ITEMS:
            found[active].append(item)

    return Steer(found["more"], found["less"])


def _cache_path(cfg: Config):
    return cfg.data_dir / "profile_steer.npz"


def steer_vectors(
    conn: sqlite3.Connection, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the steering sections, as (positive, negative) unit vectors.

    Cached against a hash of the text, because loading the encoder costs
    seconds and the profile changes far less often than recommendations are
    generated.
    """
    profile = current(conn)
    steer = parse(profile.content if profile else None)
    if not steer:
        return _empty(), _empty()

    key = steer.key(cfg.embedding_model)
    cached = _load_cache(cfg, key)
    if cached is not None:
        return cached

    from advisor.embed import encoder

    vectors = store.normalize(encoder.encode(steer.more + steer.less, cfg))
    more = vectors[: len(steer.more)]
    less = vectors[len(steer.more) :]

    _save_cache(cfg, key, more, less)
    return more, less


def _empty(dims: int = 0) -> np.ndarray:
    return np.empty((0, dims), dtype=store.DTYPE)


def _load_cache(cfg: Config, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _cache_path(cfg)
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if str(data["key"]) != key:
                return None
            return data["more"], data["less"]
    except (OSError, KeyError, ValueError):
        # A corrupt or half-written cache is not worth failing a run over.
        return None


def _save_cache(cfg: Config, key: str, more: np.ndarray, less: np.ndarray) -> None:
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=np.array(key), more=more, less=less)


def current(conn: sqlite3.Connection) -> Profile | None:
    row = conn.execute(
        "SELECT * FROM profile_versions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return Profile(row["id"], row["content"], row["written_by"], row["created_at"])


def history(conn: sqlite3.Connection, limit: int = 20) -> list[Profile]:
    return [
        Profile(row["id"], row["content"], row["written_by"], row["created_at"])
        for row in conn.execute(
            "SELECT * FROM profile_versions ORDER BY id DESC LIMIT ?", (limit,)
        )
    ]


def save(conn: sqlite3.Connection, content: str, written_by: str = "user") -> int:
    cursor = conn.execute(
        "INSERT INTO profile_versions (content, written_by, created_at) VALUES (?,?,?)",
        (content.strip(), written_by, now()),
    )
    return int(cursor.lastrowid)


def feedback_since_last_profile(conn: sqlite3.Connection) -> int:
    """How many *papers* you have rated since the profile was written.

    Distinct papers, not feedback rows. Feedback is append-only so you can
    change your mind, which means rating one paper three times writes three
    rows — counting those would report three new pieces of evidence when there
    is one, and clicking the same thumb twice would appear to age the profile.
    """
    profile = current(conn)
    if profile is None:
        return conn.execute("SELECT count(DISTINCT paper_id) FROM feedback").fetchone()[0]

    return conn.execute(
        "SELECT count(DISTINCT paper_id) FROM feedback WHERE created_at > ?",
        (profile.created_at,),
    ).fetchone()[0]


def briefing(conn: sqlite3.Connection) -> str:
    """Everything needed to have a profile written, as one block of text.

    The advisor never calls a model, so this is the handover: copy it into
    whatever assistant you already pay for — Claude on the web, say — and paste
    the answer back at ``/profile``. You get the written profile without this
    program holding a credential or spending anything, and the evidence stays
    something you can read before you send it anywhere.
    """
    evidence = gather_evidence(conn)
    if not evidence.strip():
        return ""
    return f"{BRIEF}\n---\n\n{evidence}\n"


def gather_evidence(conn: sqlite3.Connection) -> str:
    """Everything a profile could be written from, as plain text.

    Notes come last and are labelled as the strongest signal, because they are:
    a title says what a paper is, a note says what you thought of it.
    """
    parts: list[str] = []

    read = conn.execute(
        """SELECT p.title, p.categories FROM library l
             JOIN papers p ON p.id = l.paper_id
            WHERE l.status = 'read'
            ORDER BY l.added_at DESC LIMIT ?""",
        (MAX_LIBRARY_SAMPLE,),
    ).fetchall()
    if read:
        parts.append("# Papers they have read\n")
        parts += [f"- {row['title']}" for row in read]

    liked = conn.execute(
        """SELECT p.title FROM feedback f JOIN papers p ON p.id = f.paper_id
            WHERE f.rating > 0
              AND f.id = (SELECT max(id) FROM feedback g WHERE g.paper_id = f.paper_id)
            ORDER BY f.id DESC LIMIT ?""",
        (MAX_LIBRARY_SAMPLE,),
    ).fetchall()
    if liked:
        parts.append("\n# Papers they marked interesting\n")
        parts += [f"- {row['title']}" for row in liked]

    rejected = conn.execute(
        """SELECT p.title, f.tags FROM feedback f JOIN papers p ON p.id = f.paper_id
            WHERE f.rating < 0
              AND f.id = (SELECT max(id) FROM feedback g WHERE g.paper_id = f.paper_id)
            ORDER BY f.id DESC LIMIT ?""",
        (MAX_LIBRARY_SAMPLE,),
    ).fetchall()
    if rejected:
        parts.append("\n# Papers they rejected, with their stated reasons\n")
        for row in rejected:
            reasons = (row["tags"] or "[]").strip("[]").replace('"', "") or "no reason given"
            parts.append(f"- {row['title']}  ({reasons})")

    notes = conn.execute(
        """SELECT p.title, f.rating, f.note FROM feedback f
             JOIN papers p ON p.id = f.paper_id
            WHERE f.note IS NOT NULL AND trim(f.note) != ''
            ORDER BY f.id DESC LIMIT ?""",
        (MAX_NOTES,),
    ).fetchall()
    if notes:
        parts.append("\n# Their own words (strongest signal)\n")
        for row in notes:
            verdict = "liked" if row["rating"] > 0 else "rejected"
            parts.append(f'- on "{row["title"]}" ({verdict}): {row["note"]}')

    return "\n".join(parts) if parts else ""
