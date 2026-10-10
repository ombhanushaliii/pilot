"""Extract the organisation table, never profile links from page sidebars."""

from dataclasses import dataclass
from html.parser import HTMLParser
import random
import re
from urllib.parse import unquote, urljoin, urlsplit


@dataclass(frozen=True)
class Candidate:
    handle: str
    rating: int | None


def validate_organization_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "codeforces.com"
            or not re.fullmatch(r"/ratings/organization/\d+/?", parsed.path)
            or parsed.query or parsed.fragment):
        raise ValueError("Use https://codeforces.com/ratings/organization/<id>")
    return url.rstrip("/")


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records: list[Candidate] = []
        self.cells: list[str] = []
        self.cell: list[str] | None = None
        self.handle: str | None = None

    def _flush(self):
        if self.handle:
            rating = None
            if self.cells:
                value = self.cells[-1].strip().replace("\xa0", "")
                if re.fullmatch(r"\d+", value):
                    rating = int(value)
            self.records.append(Candidate(self.handle, rating))
        self.cells, self.cell, self.handle = [], None, None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr":
            self._flush()  # The actual site sometimes omits </tr>.
        elif tag in {"td", "th"}:
            self.cell = []
        elif tag == "a":
            href = attributes.get("href", "")
            match = re.fullmatch(r"/profile/([^/?#]+)", href)
            if match:
                handle = unquote(match.group(1))
                if re.fullmatch(r"[A-Za-z0-9_.-]+", handle):
                    self.handle = handle

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.cells.append("".join(self.cell))
            self.cell = None
        elif tag == "tr":
            self._flush()


def parse_organization_page(html: str, source_url: str) -> tuple[list[Candidate], list[str]]:
    """Select only the first table inside the ratingsDatatable container."""
    validate_organization_url(source_url)
    marker = re.search(r'<div\b[^>]*\bclass=["\'][^"\']*\bratingsDatatable\b[^"\']*["\']', html)
    if not marker:
        raise ValueError("Organisation rating table absent; refusing to extract arbitrary profiles")
    table = re.search(r"<table\b[^>]*>(.*?)</table\s*>", html[marker.end():], re.S | re.I)
    if not table:
        raise ValueError("Organisation rating table incomplete")
    parser = _Rows()
    parser.feed(table.group(0))
    parser._flush()
    unique = {item.handle.casefold(): item for item in parser.records}
    if not unique:
        raise ValueError("Organisation table has no handles")
    # If this page ever becomes paginated, never silently claim full coverage.
    org_path = urlsplit(source_url).path.rstrip("/")
    links = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        absolute = urljoin(source_url, href)
        p = urlsplit(absolute)
        if p.netloc == "codeforces.com" and p.path.startswith(org_path + "/page/"):
            links.append(absolute)
        elif p.netloc == "codeforces.com" and p.path.rstrip("/") == org_path and "page=" in p.query:
            links.append(absolute)
    return list(unique.values()), sorted(set(links))


def select_sample(candidates: list[Candidate], size: int, seed: int) -> list[Candidate]:
    """Deterministic random sample, independent of allegation or behaviour."""
    ordered = sorted(candidates, key=lambda x: x.handle.casefold())
    if not 1 <= size:
        raise ValueError("sample_size must be positive")
    return sorted(random.Random(seed).sample(ordered, min(size, len(ordered))),
                  key=lambda x: x.handle.casefold())

