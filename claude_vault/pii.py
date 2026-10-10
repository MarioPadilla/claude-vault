import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import requests

from .config import load_config
from .models import Conversation

# Risk level ordering for comparison
_RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _luhn_valid(candidate: str) -> bool:
    """Luhn checksum, so arbitrary 16-digit ids are not reported as cards"""
    digits = [int(c) for c in candidate if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _looks_random(token: str) -> bool:
    """Heuristic for an unprefixed secret: random-looking, not a hash or name.

    Hex digests (git SHAs, md5, sha256, uuids without dashes) top out at 4 bits
    of entropy per character, so a threshold just above that excludes them.
    Subresource-integrity hashes and slugs or identifiers built from several
    plain lowercase words ("get_user_profile_by_id_2024") are excluded too.
    """
    if re.match(r"(?:sha(?:1|256|384|512)|md5)-", token):
        return False
    if sum(c.isdigit() for c in token) < 2 or not any(c.isalpha() for c in token):
        return False
    words = [w for w in re.split(r"[_\-]", token) if re.fullmatch(r"[a-z]{3,}", w)]
    if len(words) >= 3:
        return False
    counts = Counter(token)
    entropy = -sum(n / len(token) * math.log2(n / len(token)) for n in counts.values())
    return entropy > 4.1


_NON_SECRET_LITERALS = {"null", "none", "nil", "true", "false", "undefined"}


def _plausible_credential_value(value: str) -> bool:
    """Reject values of `password = ...` that are obviously not a secret"""
    if value.lower() in _NON_SECRET_LITERALS:
        return False
    # Reference to an environment variable or template, not the secret itself
    if value[0] in "$%":
        return False
    # Attribute access such as settings.SECRET_KEY or self.token
    if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", value):
        return False
    # Short numbers are counts and limits (token_count: 1200), not secrets
    if value.isdigit() and len(value) < 6:
        return False
    return True


@dataclass
class PIIScanResult:
    """Result of a PII scan on a conversation"""

    detected: bool
    risk_level: str  # "none" | "low" | "medium" | "high"
    pii_types: List[str] = field(default_factory=list)
    # type → list of raw matched strings (used for redaction)
    matches: Dict[str, List[str]] = field(default_factory=dict)
    # True only when LLM classification actually ran and returned a verdict
    llm_used: bool = False


class PIIDetector:
    """Detects and redacts PII in conversation text using regex and optional LLM."""

    # Compiled regex patterns keyed by PII type. Order matters for redaction:
    # a pattern that swallows a larger span must run before ones it contains.
    REGEX_PATTERNS: Dict[str, re.Pattern] = {
        # The whole block; or, when it was truncated, the header plus the base64
        # lines after it. A bare header mentioned in prose is left alone.
        "private_key": re.compile(
            r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----"
            r"(?:[\s\S]{0,8192}?-----END [A-Z ]{0,30}PRIVATE KEY-----"
            r"|(?:\s*[A-Za-z0-9+/=]{16,}){1,200})"
        ),
        # scheme://user:password@host, before email so its user:pass@ is not
        # reported as an address
        "connection_string": re.compile(
            r"\b[a-zA-Z][a-zA-Z0-9+.\-]{0,30}://[^\s:/@]*:[^\s@/]+@"
        ),
        # Length caps are the RFC limits; unbounded runs backtrack quadratically
        "email": re.compile(
            r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,24}\b"
        ),
        # Needs a separator, parentheses or a leading +, so bare 10-digit ids and
        # epoch timestamps are not phone numbers
        "phone": re.compile(
            r"(?<![\w.])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\w)"
        ),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
        "jwt": re.compile(
            r"\beyJ[A-Za-z0-9_\-]{8,256}\.eyJ[A-Za-z0-9_\-]{8,2048}\.[A-Za-z0-9_\-]{0,1024}"
        ),
        # Provider-specific token formats
        "api_key": re.compile(
            r"\b(?:"
            r"sk-(?:ant|proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}"  # Anthropic, OpenAI
            r"|sk-[A-Za-z0-9]{20,}"  # OpenAI (legacy)
            r"|gh[pousr]_[A-Za-z0-9]{36,}"  # GitHub tokens
            r"|github_pat_[A-Za-z0-9_]{22,}"  # GitHub fine-grained PAT
            r"|AIza[A-Za-z0-9\-_]{35}"  # Google API key
            r"|(?:AKIA|ASIA)[0-9A-Z]{16}"  # AWS access key id
            r"|xox[abprs]-[A-Za-z0-9\-]{10,}"  # Slack
            r"|[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe
            r"|glpat-[A-Za-z0-9_\-]{20,}"  # GitLab
            r"|hf_[A-Za-z0-9]{34,}"  # Hugging Face
            r"|npm_[A-Za-z0-9]{36}"  # npm
            r"|ya29\.[A-Za-z0-9_\-]{20,}"  # Google OAuth
            r"|SG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"  # SendGrid
            r")\b"
            r"|(?i:\b(?:Bearer|Basic))\s+[A-Za-z0-9\-._~+/]{16,}=*"  # Authorization
        ),
        "ip_address": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        # name = value where the name says it is a secret. Only the value is
        # replaced (the named group), so the name and any delimiters survive.
        # An unquoted value must end at a delimiter, which rules out calls and
        # expressions such as `token = self.next()`.
        "credential_context": re.compile(
            r"(?i)(?:password|passwd|secret|token|api[_\-]?key|access[_\-]?key)"
            r"(?:[_\-][\w\-]{0,40})?[\"']?\s*[:=]\s*"
            r"(?:\"(?P<dq>[^\"\n]{4,})\""
            r"|'(?P<sq>[^'\n]{4,})'"
            r"|(?P<raw>[^\s\"'`,;(){}\[\]<>]{4,})(?=[\s\"'`,;)}\]>]|\Z))"
        ),
        # Last: anything long and random-looking that no rule above recognised.
        # Not part of a base64 run, which the pattern would otherwise cut up.
        "secret_token": re.compile(
            r"(?<![A-Za-z0-9_\-+/])[A-Za-z0-9_\-]{32,}(?![A-Za-z0-9_\-+/=])"
        ),
    }

    # Extra check on a regex match, to cut false positives the pattern cannot
    _VALIDATORS: Dict[str, Callable[[str], bool]] = {
        "credit_card": _luhn_valid,
        "credential_context": _plausible_credential_value,
        "secret_token": _looks_random,
    }

    # Minimum risk level assigned to each PII type
    _TYPE_RISK: Dict[str, str] = {
        "ssn": "high",
        "credit_card": "high",
        "api_key": "high",
        "jwt": "high",
        "private_key": "high",
        "connection_string": "high",
        "email": "medium",
        "phone": "medium",
        "secret_token": "medium",
        "ip_address": "low",
        "credential_context": "low",
    }

    def __init__(self) -> None:
        self.config = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _find_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Locate sensitive spans as (start, end, type), in text order.

        Patterns run in order, and each span found is blanked before the next
        pattern runs. A later pattern therefore cannot re-report part of it (the
        user:pass@ of a connection string is not also an email address), and
        scan() and redact() always agree on what was found.
        """
        spans: List[Tuple[int, int, str]] = []
        remaining = text

        for pii_type, pattern in self.REGEX_PATTERNS.items():
            valid = self._VALIDATORS.get(pii_type)
            found: List[Tuple[int, int]] = []
            for m in pattern.finditer(remaining):
                # A pattern may name the exact part to replace, else the match
                start, end = m.span(m.lastgroup) if m.lastgroup else m.span()
                if valid is None or valid(remaining[start:end]):
                    found.append((start, end))
            if not found:
                continue

            spans.extend((start, end, pii_type) for start, end in found)
            parts, pos = [], 0
            for start, end in found:
                parts.append(remaining[pos:start])
                parts.append(" " * (end - start))
                pos = end
            parts.append(remaining[pos:])
            remaining = "".join(parts)

        return sorted(spans)

    def scan(self, text: str) -> PIIScanResult:
        """Run regex-only scan on a block of text."""
        matches: Dict[str, List[str]] = {}
        for start, end, pii_type in self._find_spans(text):
            matches.setdefault(pii_type, []).append(text[start:end])

        detected = bool(matches)
        risk_level = self._compute_risk(list(matches.keys()))

        return PIIScanResult(
            detected=detected,
            risk_level=risk_level,
            pii_types=list(matches.keys()),
            matches=matches,
        )

    def redact(self, text: str) -> str:
        """Replace all detected PII in *text* with [REDACTED-TYPE] placeholders."""
        parts, pos = [], 0
        for start, end, pii_type in self._find_spans(text):
            parts.append(text[pos:start])
            parts.append(f"[REDACTED-{pii_type.upper().replace('_', '-')}]")
            pos = end
        parts.append(text[pos:])
        return "".join(parts)

    def classify_with_llm(self, conversation: Conversation) -> Dict:
        """
        Use Ollama to classify whether a conversation contains sensitive content.

        Returns a dict with keys: is_sensitive, risk_level, reason, classified.
        Falls back to {"is_sensitive": False, "risk_level": "none"} on any error,
        with classified False so callers can tell a fallback from a real verdict.
        """
        sample = self._build_sample(conversation)
        prompt = (
            "You are a privacy analyst. Analyse the following conversation excerpt "
            "and determine whether it contains sensitive or confidential information "
            "(passwords, API keys, medical records, financial data, confidential "
            "business information, personal identity details, etc.).\n\n"
            f"Conversation excerpt:\n{sample}\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            '{"is_sensitive": true/false, "risk_level": "low|medium|high|none", '
            '"reason": "brief explanation"}'
        )

        try:
            response = requests.post(
                self.config.ollama.url,
                json={
                    "model": self.config.ollama.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                },
                timeout=self.config.ollama.timeout,
            )
            response.raise_for_status()
            raw = response.json().get("response", "")
            # Extract the JSON object from the response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                risk = data.get("risk_level", "none")
                if risk not in _RISK_ORDER:
                    risk = "none"
                return {
                    "is_sensitive": bool(data.get("is_sensitive", False)),
                    "risk_level": risk,
                    "reason": data.get("reason", ""),
                    "classified": True,
                }
        except Exception:
            pass

        return {
            "is_sensitive": False,
            "risk_level": "none",
            "reason": "",
            "classified": False,
        }

    def analyze(
        self, conversation: Conversation, use_llm: bool = True
    ) -> PIIScanResult:
        """
        Full analysis: regex scan across all messages, optionally merged with LLM
        classification.

        Args:
            conversation: The conversation to analyse.
            use_llm: If True, also run LLM classification (requires Ollama).

        Returns:
            PIIScanResult with the highest risk level from either method.
        """
        # Aggregate all message content
        full_text = "\n".join(m.content for m in conversation.messages)
        result = self.scan(full_text)

        if use_llm and self._ollama_available():
            llm_result = self.classify_with_llm(conversation)
            result.llm_used = llm_result.get("classified", False)
            llm_risk = llm_result.get("risk_level", "none")
            # Elevate risk level if LLM found something higher
            if _RISK_ORDER.get(llm_risk, 0) > _RISK_ORDER.get(result.risk_level, 0):
                result.risk_level = llm_risk
                result.detected = True
                if "sensitive" not in result.pii_types:
                    result.pii_types.append("sensitive")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_risk(self, pii_types: List[str]) -> str:
        if not pii_types:
            return "none"
        return max(
            (self._TYPE_RISK.get(t, "low") for t in pii_types),
            key=lambda r: _RISK_ORDER[r],
        )

    def _ollama_available(self) -> bool:
        try:
            base_url = self.config.ollama.url.rsplit("/api/", 1)[0]
            response = requests.get(base_url, timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def _build_sample(self, conversation: Conversation, max_chars: int = 2000) -> str:
        """Build a representative text sample from a conversation for LLM analysis."""
        parts = [f"Title: {conversation.title}"]
        char_budget = max_chars - len(parts[0])
        for msg in conversation.messages:
            snippet = msg.content[:500]
            if char_budget <= 0:
                break
            parts.append(snippet)
            char_budget -= len(snippet)
        return "\n".join(parts)
