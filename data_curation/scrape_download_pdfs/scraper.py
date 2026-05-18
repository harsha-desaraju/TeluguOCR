
import trafilatura
from urllib.parse import urlparse, urljoin, unquote
from bs4 import BeautifulSoup
import httpx
import time
import pypdf
import io
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field


@dataclass
class FetchedResource:
    url: str
    content: bytes
    content_type: str   # "text/html", "application/json", ...
    status: int


@dataclass
class PageResult:
    url: str
    html: str
    text: Union[str, None]
    page_links: list[str] = field(default_factory=list)
    file_links: list[str] = field(default_factory=list)

@dataclass
class FetchedFile:
    url: str
    local_path: Path
    kind: str
    data: Union[object, None]

@dataclass
class ScrapePolicy:
    allowed_exts: set =  field(default_factory=lambda: {"", ".html", ".htm", ".php", ".aspx"})
    data_exts: set = field(default_factory=lambda: {".pdf"})
    same_domain_only: bool = True


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (web-agent)",
    "Accept": "*/*",
}




class WebScraper:
    def __init__(
        self,
        out_dir: Union[str, Path] = "scraped",
        timeout: float = 20.0,
        policy: ScrapePolicy = ScrapePolicy(),
        target_prefix: str = ""
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.target_prefix = target_prefix
        self.client = httpx.Client(
            headers=DEFAULT_HEADERS, timeout=timeout, follow_redirects=True
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.close()

    def fetch(self, url: str) -> FetchedResource:
        r = self.client.get(url)
        r.raise_for_status()
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        return FetchedResource(url=str(r.url), content=r.content, content_type=ct, status=r.status_code)


    def extract_page(self, url: str) -> PageResult:
        res = self.fetch(url)
        html = res.content.decode(errors="replace")

        text = None
        if trafilatura is not None:
            text = trafilatura.extract(
                html,
                favor_precision=False,
                no_fallback=False,
                include_tables=True,
                target_language="te"
            )

        page_links: list[str] = []
        for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
            absu = urljoin(url, a["href"])
            parsed = urlparse(absu)
            if parsed.scheme not in ("http", "https"):
                continue
            page_links.append(absu)

        page_links = self._dedupe(page_links)
        page_links, file_links = self._apply_policy(page_links, urlparse(url).netloc)

        return PageResult(
            url=res.url, html=html, text=text,
            page_links=page_links,
            file_links=file_links
        )

    @staticmethod
    def _safe_name(url: str) -> str:
        return urlparse(url).path.replace("/", "_").strip("_") or "index"

    @staticmethod
    def _parse_pdf(b: bytes) -> str:
        if pypdf is None:
            raise RuntimeError("pip install pypdf to parse PDF")
        reader = pypdf.PdfReader(io.BytesIO(b))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)

    def download(self, url: str, overwrite: bool = False) -> FetchedFile:
        name = Path(urlparse(url).path).name or self._safe_name(url)
        local = self.out_dir / name

        if not local.exists() or overwrite:
            res = self.fetch(url)
            local.write_bytes(res.content)
            # time.sleep(self.delay)

        kind = local.suffix.lstrip(".").lower() or "other"
        data = None
        try:
            data = self._parse_pdf(local.read_bytes())
        except Exception as e:
            print(f"[warn] parse failed for {local.name}: {e}")
        return FetchedFile(url=url, local_path=local, kind=kind, data=data)

    @staticmethod
    def _dedupe(xs):
        seen = set()
        return [x for x in xs if not (x in seen or seen.add(x))]

    def _apply_policy(self, page_links: list[str], host: str) -> tuple[list[str], list[str]]:
        picked_links: list[str] = []
        picked_files: list[str] = []

        for link in page_links:
            parsed = urlparse(unquote(link))
            path = parsed.path
            ext = Path(parsed.path).suffix.lower()

            domain_ok = (not self.policy.same_domain_only) or (parsed.netloc == host)

            if domain_ok and path.startswith(self.target_prefix):
                if ext in self.policy.allowed_exts:
                    picked_links.append(link)
                elif ext in self.policy.data_exts:
                    picked_files.append(link)

        return picked_links, picked_files





if __name__ == '__main__':

    base_url = "https://ebooks.tirumala.org/read?id=245&title="

    print(unquote(base_url))

    with WebScraper(out_dir="generic_scraper/pdf_files", timeout=60.0) as scraper:

        result = scraper.extract_page(base_url)

        print(result.text)
        print('='*100)
        print(result.file_links)
        print(len(result.file_links))
        print(f"Download started!")
        scraper.download(result.file_links[0])
        print(f"Download finished!")
