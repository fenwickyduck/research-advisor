# Research Advisor

A personal reading advisor for research papers. You tell it what you have read,
it tells you what to read next, you tell it what you thought, and it gets
better at it.

It runs entirely on your own machine. No account, no API key, no service to
sign up for, nothing sent anywhere except the public archives it harvests from.

## How it works

Four moving parts, and it is worth understanding them because every command
below belongs to one of them.

**1. A corpus.** It downloads the metadata — title, abstract, authors — of
every paper on arXiv `cs.CR` and the Cryptology ePrint Archive: about 76,000 of
them. This is *harvesting*, and it is incremental, so after the first time it
only fetches what changed.

**2. Embeddings.** Each paper's title and abstract are run through
[SPECTER](https://huggingface.co/allenai/specter), a model trained on
scientific text, which turns them into a 768-number vector. Papers about
similar things end up with similar vectors. This is *embedding*, it happens
locally on your CPU, and it is the slow part — about three hours for the whole
corpus, or seconds if you load a prebuilt snapshot.

**3. Your library.** The papers you say you have read, plus what you thought of
them. Averaging their vectors gives a picture of your interests — several
pictures, in fact, since someone reading across two subfields is badly
described by one average.

**4. Retrieval.** To recommend, it compares your interest vectors against every
paper in the corpus, drops near-duplicates so a batch is not eight variations
on one result, excludes anything you have already seen, and adds new work by
authors you follow. Each recommendation is labelled with the reason it was
chosen — the paper of yours it is nearest to, or the author you follow.

There is no language model anywhere in that loop, which is why it costs nothing
to run and why your reading never leaves the machine.

## Setup

Needs Python 3.11+ and about 2 GB of disk.

```sh
git clone https://github.com/fenwickyduck/research-advisor.git
cd research-advisor
./install.sh
```

That is the whole thing. It creates the virtualenv, installs, and — if you have
the [GitHub CLI](https://cli.github.com) logged in — downloads the shared corpus
snapshot, so you skip an hour of harvesting and three hours of encoding. Safe to
re-run; every step checks whether it is already done.

Then:

```sh
source .venv/bin/activate          # once per terminal
advisor add 2401.12345 2024/123    # papers you have read — ten is plenty
advisor serve                      # http://127.0.0.1:8000
```

**Activating the virtualenv is what puts `advisor` on your `PATH`.** Without it
you get `command not found`, because the command lives in `.venv/bin/`. If you
would rather not activate anything, every command also works spelled out —
`.venv/bin/advisor status`. That is the form for scripts and cron, which have no
shell to activate in; `advisor schedule` and `advisor mcp --config` already print
absolute paths for that reason.

### Doing it by hand

If you would rather not run a script, or you want to build the corpus yourself
instead of trusting a prebuilt file:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'

advisor snapshot fetch             # download the shared corpus (~151 MB), or:
advisor harvest                    # build it yourself — tens of minutes
advisor embed                      # ~3 hours on a laptop CPU
```

You do not have to wait for `advisor embed` to finish. It encodes your own
library first and then spreads across years, so recommendations start working
within a couple of minutes and simply get better. Stop and rerun it whenever —
it resumes where it left off.

## Commands

Everything is `advisor <command>`; add `--help` to any of them.

### Getting papers in

| | |
|---|---|
| `advisor add <ids…>` | Record papers you have read. Takes arXiv IDs, ePrint IDs, DOIs, or any URL containing one — as arguments, or one per line on stdin. Also at `/add`. |
| `advisor harvest` | Download new paper metadata from arXiv and ePrint. Incremental: it remembers where it got to, so later runs fetch only what changed. Safe to interrupt. |
| `advisor embed` | Turn papers into vectors. Resumable, and ordered so partial results are useful. Nothing can be recommended until it is embedded. |
| `advisor snapshot fetch` | Download the shared corpus and load it — 76,000 papers, already embedded, in about a minute. Needs `gh` logged in, since the repository is private. |
| `advisor snapshot save\|load\|show <file>` | Write, restore or inspect that file yourself. |

### Getting papers out

| | |
|---|---|
| `advisor recommend` | What to read next. Records the batch, as the web feed does, so nothing is suggested twice. `--preview` looks without recording; `-n` sets how many. |
| `advisor search <query>` | Full-text search over the corpus. Quote a phrase to keep it together; prefix a word with `-` to exclude it. |
| `advisor serve` | The web UI at `http://127.0.0.1:8000` — feed, library, search, authors, profile, data. |
| `advisor status` | What is in the database and how far embedding has got. Start here when something seems wrong. |

### Teaching it about you

| | |
|---|---|
| `advisor profile` | Show your interest profile. `--brief` prints your reading and ratings with instructions, to hand to an assistant of your choosing. |
| `advisor follow <name>` | Follow an author; their new work appears whatever it is about. Refuses a name nobody in your corpus wrote, and suggests near matches. No arguments lists who you follow; `--suggest` names authors you have read more than once. |
| `advisor unfollow <name>` | Stop following someone. |
| `advisor mcp` | Serve your library to an assistant over MCP. `--config` prints the setup command. |

### Housekeeping

| | |
|---|---|
| `advisor update` | Harvest, embed within a time budget, and build a fresh feed — the whole nightly job in one command. |
| `advisor schedule` | Print a systemd timer and a crontab line for `advisor update`. Installs nothing. |
| `advisor export <file>` | Write your library, ratings, notes, profile and follows to a file. |
| `advisor import <file>` | Load that file into any install. Merges by default; `--replace` overwrites. |
| `advisor reset` | Forget what it learned about you, keeping the corpus. `--all` empties everything. |

## The parts worth explaining

### Feedback

A thumbs-down opens optional reason chips and a note box. The reason is not
decoration — it changes how far the rejection moves your interests:

| reason | effect on retrieval |
|---|---|
| I already know this | none — retrieval was right, the paper is just redundant |
| Wrong subfield | full push away |
| Too theoretical / applied / not rigorous | half push |
| Want newer work | leaves interests alone, leans the recency prior (capped at 3x) |

An unlabelled thumbs-down gets the full default push, so explaining yourself
stays optional.

### The interest profile

Ranking is similarity, diversification and attribution — no model call. But
embeddings only capture "more like this". Only words carry "I already know RLWE
hardness, stop showing me introductions".

So you write a profile, at `/profile`, and **two of its sections are
instructions rather than description**:

```markdown
## More of
doubly-efficient private information retrieval
zero-knowledge proof systems and succinct arguments

## Less of
post-quantum migration policy surveys
```

Each line is encoded by the same local model that encodes the corpus — SPECTER
takes arbitrary text, not only papers — so a line under **More of** becomes a
query vector, and one under **Less of** becomes something to steer away from,
at the same strength as a "wrong subfield" rejection.

Write them like paper titles, not sentences about yourself. That is what the
encoder was trained on, and it shows: the top corpus matches for `lattice-based
digital signatures` are CRYSTALS-Dilithium, *Sharper Ring-LWE Signatures* and
*Asymptotically Efficient Lattice-Based Digital Signatures*.

A stated interest is one interest among your others, not an override. A profile
is also enough on its own — with an empty library, saying what you want is a
complete starting point. Everything under other headings is prose, for you to
read; `/profile` tells you which of your lines are actually steering anything.

### Following people

Embeddings cannot answer "what has this group published since", and a followed
author's next paper may not resemble anything in your library — which is often
exactly why you want it. So follows are retrieved **by name, outside the vector
search**, and lead each batch:

```
[  --  ] Topology-Hiding Computation From Key Agreement in Diameter-Two…
         By Shafi Goldwasser, whom you follow.
[0.915] Cryptanalysis of a (Somewhat) Additively Homomorphic Encryption…
         Closest to "Fully Homomorphic Encryption over the Integers", which you read.
```

They carry no similarity score, because similarity had no part in choosing them.

Names match on surname plus given name, accents folded, so "Henry
Corrigan-Gibbs" and "H. Corrigan-Gibbs" are one person. Initials are treated as
initials only when one side actually abbreviates: **"Wei Chen" matches "W. Chen"
but not "Wen Chen"**. Reducing both to `chen|w` — the obvious first
implementation — turned one followed author into four on a corpus this size.

### Talking to an assistant about it

`advisor mcp` serves your library, the corpus and your profile over the Model
Context Protocol, so an assistant you already run can consult them. **The
direction is the opposite of the usual integration: the advisor never calls a
model.** It answers questions, holds no credential, and speaks over a pipe to a
client that launches it, so nothing listens on a port.

On Linux the client is **Claude Code**, a CLI (Claude Desktop is macOS and
Windows only):

```sh
claude mcp add research-advisor -- "$PWD/.venv/bin/advisor" mcp
claude mcp list          # should say ✔ Connected
claude                   # then ask "what have I been reading?"
```

`advisor mcp --config` prints that with the right absolute path, plus the JSON
form for other clients. MCP servers load at startup, so an already-open session
will not see it.

Three prompts, appearing as slash commands:

| | |
|---|---|
| `/refresh_profile` | rewrite the profile from everything read and rated since |
| `/explore <topic>` | work out what a direction is really called, then steer toward it |
| `/review_feed` | go through the current batch and say which of it is noise |

They are not templates — each reads the database first and arrives carrying the
evidence, and `/explore` searches the corpus before asking anything, so the
discussion starts from real titles rather than your guess at the vocabulary.
Of the twelve tools, nine are read-only; the three that write are the profile
and following, each annotated so the client asks first. Nothing can delete a
paper or reset the database.

If you would rather not set that up, `advisor profile --brief` prints the same
evidence and instructions as text, to paste into whatever assistant you like.

### Why every recommendation comes with a reason

```
3. [0.943] (2026) GPU Acceleration of Learning With Errors KEMs Using OpenACC
      Closest to "Portable Acceleration of Lattice KEMs for
      Post-Quantum Cryptography", which you read (cosine 0.95).
```

Naming the paper that pulled a recommendation in is the honest answer to "why
am I being shown this" — it is literally what drove the match, it costs one
matrix product, and it is easier to check than a written justification, because
you recognise the paper it names. Papers you have explicitly disliked are never
offered as the reason for a new one.

### How it stays current

Both sources are harvested over OAI-PMH, whose `from` parameter filters on
**modification** date rather than submission date. That matters more than it
sounds:

- **Revisions are picked up.** When an author posts a v2 with a rewritten
  abstract, the stored abstract is replaced. Since the abstract is what gets
  embedded, harvesting on submission date would leave recommendations computed
  from text the author had already replaced.
- **Late cross-listings appear.** A paper moved into `cs.CR` months after
  submission shows up on the next run.
- **Withdrawals are recorded.** Deleted records set `withdrawn_at`, so the paper
  stops being recommended — but the row survives, because a paper already in
  your library should not vanish from your history.

Merging follows two rules. A record arriving again **from the same source** is
authoritative and replaces the stored content. A record matched **across
sources** (same title and an overlapping author) fills in gaps and adds its
identifier, but never overwrites a fuller record with a terser one — and
category lists always union, so a paper on both archives keeps both taxonomies.

## Your data, and sharing this

Your library, ratings, notes, profile and follows live in
`~/.local/share/advisor/`. **They are not in this repository and never go into
it** — every example above is invented.

```sh
advisor export mine.json     # take it to another machine, or keep as a backup
advisor import mine.json     # merges by default
```

The same pair is at `/data` in the browser. Paper ids are local numbers, so an
export carries each remembered paper's real identifiers *and* enough metadata to
recreate it — an import works against an empty database, before anything has
been harvested. It also carries which papers you have already been shown, so a
move does not re-offer everything you looked at once and passed over.
Importing twice changes nothing.

The corpus travels separately, with `advisor snapshot`, because it is the
opposite kind of thing: large, public, identical for everyone, and expensive to
rebuild.

```sh
advisor snapshot save corpus.tar    # ~151 MB, about 5 seconds
advisor snapshot load corpus.tar    # about 15 seconds, versus 3 hours of encoding
```

Vectors ship as float16, which halves 223 MB to 112 MB: they are unit vectors,
and against the float32 originals the top-10 and top-50 neighbours come back
identical, with a largest cosine error of 3e-5. The metadata is gzipped JSON
Lines, 134 MB down to 42 MB. The result is too big for a file in a git
repository (100 MB limit) and belongs in a **release asset** (2 GB limit)
anyway, being a build artifact rather than source.

A snapshot contains no library, ratings, notes or profile — the tests assert it.

## Running unattended

`advisor update` is the whole nightly job. The three steps belong together:
harvesting without embedding leaves new papers invisible to retrieval, and
embedding without a run leaves them out of the feed.

- **The embed pass is bounded** (`--max-minutes`, default 30), so the first run
  after a backfill cannot hold the machine for hours; the remainder is picked up
  the following night.
- **Passes cannot overlap.** A manual `advisor embed` and a scheduled `advisor
  update` take an advisory lock, because both rewrite the vector matrix
  wholesale and the loser would silently discard the winner's work.

`advisor schedule` prints a ready-made systemd timer and crontab line pointing
at this installation's absolute path. It installs nothing.

## Configuration

Settings, if you want any, go in `~/.config/advisor/config.toml`. See
`advisor/config.py` for the keys and defaults — which categories to harvest, how
far back, how many recommendations, how hard a "More of" line pulls.

## Tests

```sh
.venv/bin/python -m pytest
```

No network required: parser tests run against saved API responses in
`tests/fixtures/`. `tests/test_no_api.py` asserts that no model SDK is imported
anywhere, that no credential is read from the environment, and that the
recommendation pipeline has no seam a client could be passed through.

## Attribution

Paper metadata comes from the [arXiv API](https://arxiv.org), the
[Cryptology ePrint Archive](https://eprint.iacr.org) (© IACR and the respective
authors, harvested via OAI-PMH under its stated terms), and
[Crossref](https://www.crossref.org). A corpus snapshot redistributes that
metadata; check those terms before publishing one broadly.

Licensed under the MIT License — see [LICENSE](LICENSE).
