import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make every real HTTP request fail, as it does when Ollama is not running.

    Sync, tagging and PII analysis all probe http://localhost:11434. Left alone,
    that makes a test depend on whether Ollama happens to be running on the
    machine (with it up, the suite would make real model calls), and on Windows a
    refused connection to localhost takes seconds to fail, which turned a 6 second
    suite into a 5 minute one. Tests that need a response patch requests.get or
    requests.post themselves, which takes precedence over this.
    """

    def refuse(self, method, url, *args, **kwargs):
        raise requests.exceptions.ConnectionError(f"network disabled in tests: {url}")

    monkeypatch.setattr(requests.Session, "request", refuse)
