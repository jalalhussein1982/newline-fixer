from fastapi.testclient import TestClient

from newline_fixer.models.registry import get_fixer
from newline_fixer.service.app import create_app
from newline_fixer.service.config import Settings


def test_index_page_is_served() -> None:
    with TestClient(create_app(Settings(model="rules"), loader=lambda _s: get_fixer("rules"))) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "/v1/fix" in r.text and "<textarea" in r.text
        assert "<script src=" not in r.text  # no external assets; the page works offline
