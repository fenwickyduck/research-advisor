"""Parser tests against saved API responses. No network."""

from __future__ import annotations

import json

from advisor.ingest import arxiv, crossref, eprint, oai

from .conftest import fixture


def test_arxiv_single_entry() -> None:
    page = arxiv.parse_feed(fixture("arxiv_entry.xml"))
    assert len(page.papers) == 1

    paper = page.papers[0]
    assert paper.title == "Attention Is All You Need"
    assert paper.arxiv_id == "1706.03762"  # version suffix stripped
    assert paper.authors[0] == "Ashish Vaswani"
    assert len(paper.authors) == 8
    assert paper.published_at == "2017-06-12"
    assert "cs.CL" in paper.categories
    assert paper.url == "https://arxiv.org/abs/1706.03762"
    assert paper.pdf_url and paper.pdf_url.startswith("http")
    # The abstract is unwrapped from arXiv's hard line breaks.
    assert paper.abstract and "\n" not in paper.abstract
    assert paper.abstract.startswith("The dominant sequence transduction models")


def test_arxiv_page_of_results() -> None:
    page = arxiv.parse_feed(fixture("arxiv_page.xml"))
    assert len(page.papers) == 3
    assert all(p.title and p.arxiv_id and p.abstract for p in page.papers)
    assert all(p.published_at and len(p.published_at) == 10 for p in page.papers)
    # totalResults describes the whole query, not just this page.
    assert page.total > 40_000


def test_arxiv_total_falls_back_to_page_size_when_absent() -> None:
    minimal = """<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    assert arxiv.parse_feed(minimal).total == 0



def test_eprint_get_record() -> None:
    batch = eprint.parse_records(fixture("eprint_getrecord.xml"))
    assert len(batch.papers) == 1

    paper = batch.papers[0]
    assert paper.eprint_id == "2026/1688"
    assert paper.title.startswith("Hell")
    assert "Erik Mårtensson" in paper.authors
    assert paper.categories == ["Foundations"]
    assert paper.published_at == "2026-01-01"
    assert paper.url == "https://eprint.iacr.org/2026/1688"
    assert paper.pdf_url == "https://eprint.iacr.org/2026/1688.pdf"
    assert paper.abstract and "neural network" in paper.abstract.lower()


def test_eprint_list_records() -> None:
    batch = eprint.parse_records(fixture("eprint_listrecords.xml"))
    assert len(batch.papers) > 5
    assert all(p.eprint_id and p.title for p in batch.papers)
    # Every ID is normalised to a zero-padded four-digit sequence number.
    assert all(len(p.eprint_id.split("/")[1]) == 4 for p in batch.papers)


def test_eprint_no_records_is_not_an_error() -> None:
    """A cursor with nothing new returns noRecordsMatch, which is normal."""
    batch = eprint.parse_records(fixture("eprint_norecords.xml"))
    assert batch.papers == []
    assert batch.resumption_token is None


def test_crossref_work() -> None:
    message = json.loads(fixture("crossref_work.json"))["message"]
    paper = crossref.parse_work(message)

    assert paper is not None
    assert paper.title.startswith("Public-Key Cryptosystems")
    assert paper.authors == ["Pascal Paillier"]
    # DOIs are case-insensitive; we fold them so lookups match a pasted variant.
    assert paper.doi == "10.1007/3-540-48910-x_16"
    # Crossref reports this LNCS chapter's print date as the 2007 reissue rather
    # than the 1999 conference; we record what Crossref says, in ISO form.
    assert paper.published_at == "2007-10-10"
    assert paper.venue == "Lecture Notes in Computer Science"


def test_crossref_date_falls_back_through_available_fields() -> None:
    item = {"title": ["X"], "created": {"date-parts": [[2020, 3]]}}
    assert crossref.parse_work(item).published_at == "2020-03-01"


def test_crossref_strips_jats_markup() -> None:
    paper = crossref.parse_work(
        {
            "title": ["A paper"],
            "abstract": "<jats:p>Hello <jats:italic>world</jats:italic>.</jats:p>",
            "DOI": "10.1/x",
        }
    )
    assert paper is not None
    assert paper.abstract == "Hello world ."


# --------------------------------------------------- arXiv OAI-PMH (bulk path)


def test_set_spec_maps_category_to_arxiv_oai_set() -> None:
    assert arxiv.set_spec("cs.CR") == "cs:cs:CR"
    assert arxiv.set_spec("math.NT") == "math:math:NT"
    assert arxiv.set_spec("physics") == "physics"


def test_oai_identifier_handles_both_arxiv_id_schemes() -> None:
    assert arxiv.oai_identifier("oai:arXiv.org:2508.12220") == "2508.12220"
    assert arxiv.oai_identifier("oai:arXiv.org:cs.CR/0512345") == "cs.CR/0512345"
    assert arxiv.oai_identifier("oai:eprint.iacr.org:2024/1") is None


def test_arxiv_oai_records() -> None:
    batch = arxiv.parse_records(fixture("arxiv_oai_records.xml"))

    assert len(batch.papers) > 10
    assert all(p.arxiv_id and p.title and p.abstract for p in batch.papers)

    paper = next(p for p in batch.papers if p.arxiv_id == "2508.12220")
    # Names arrive split into forenames/keyname and are joined in reading order.
    assert paper.authors == ["Abdullah X"]
    assert paper.categories == ["cs.LG", "cs.AI", "cs.CR"]
    assert paper.published_at == "2026-08-12"
    # This record is a revision — created 08-12, updated 08-14. Harvesting on
    # the updated date is exactly what the search API could not see.
    assert paper.updated_at == "2026-08-14"
    assert paper.url == "https://arxiv.org/abs/2508.12220"


def test_arxiv_oai_falls_back_to_header_datestamp_when_never_revised() -> None:
    """A paper still at v1 has no <updated>; the header datestamp stands in."""
    xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
      <record><header>
        <identifier>oai:arXiv.org:2401.00001</identifier><datestamp>2024-01-05</datestamp>
      </header><metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2401.00001</id><created>2024-01-05</created>
          <title>A Paper</title><categories>cs.CR</categories>
          <abstract>Text.</abstract>
          <authors><author><keyname>Smith</keyname><forenames>Ann</forenames></author></authors>
        </arXiv>
      </metadata></record></ListRecords></OAI-PMH>"""
    paper = arxiv.parse_records(xml).papers[0]
    assert paper.updated_at == "2024-01-05"
    assert paper.authors == ["Ann Smith"]


# ------------------------------------------------------ OAI envelope behaviour


def test_deleted_records_are_reported_not_parsed() -> None:
    xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
      <record><header status="deleted">
        <identifier>oai:arXiv.org:2401.00002</identifier><datestamp>2024-06-01</datestamp>
      </header></record>
      <record><header>
        <identifier>oai:arXiv.org:2401.00003</identifier><datestamp>2024-06-01</datestamp>
      </header><metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2401.00003</id><created>2024-01-05</created><title>Live</title>
          <abstract>Text.</abstract><categories>cs.CR</categories>
          <authors><author><keyname>Smith</keyname></author></authors>
        </arXiv>
      </metadata></record>
      <resumptionToken>next-page</resumptionToken></ListRecords></OAI-PMH>"""

    batch = arxiv.parse_records(xml)

    assert [p.arxiv_id for p in batch.papers] == ["2401.00003"]
    assert batch.deleted == ["2401.00002"]
    assert batch.resumption_token == "next-page"


def test_oai_error_other_than_no_records_raises() -> None:
    xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <error code="badArgument">unknown set</error></OAI-PMH>"""
    try:
        arxiv.parse_records(xml)
    except oai.OAIError as exc:
        assert exc.code == "badArgument"
    else:
        raise AssertionError("expected OAIError")


def test_request_params_never_sends_from_alongside_a_token() -> None:
    """The protocol forbids it — a token already encodes the original request."""
    first = oai.request_params("arXiv", since="2024-01-01", set_spec="cs:cs:CR")
    assert first == {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "from": "2024-01-01",
        "set": "cs:cs:CR",
    }

    resumed = oai.request_params("arXiv", since="2024-01-01", set_spec="cs:cs:CR", token="t")
    assert resumed == {"verb": "ListRecords", "resumptionToken": "t"}


def test_parse_identity_normalises_granularity() -> None:
    """arXiv reports a bare date, ePrint a full timestamp; both reduce to a date."""
    arxiv_xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><Identify>
      <earliestDatestamp>2005-09-16</earliestDatestamp></Identify></OAI-PMH>"""
    eprint_xml = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><Identify>
      <earliestDatestamp>1996-01-01T00:00:00Z</earliestDatestamp></Identify></OAI-PMH>"""

    assert oai.parse_identity(arxiv_xml).earliest_datestamp == "2005-09-16"
    assert oai.parse_identity(eprint_xml).earliest_datestamp == "1996-01-01"
