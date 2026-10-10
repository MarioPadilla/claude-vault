import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

import requests

from .config import load_config
from .models import Conversation

# Risk level ordering for comparison
_RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


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

    # Compiled regex patterns keyed by PII type
    REGEX_PATTERNS: Dict[str, re.Pattern] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "phone": re.compile(
            r"\b(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})\b"
        ),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
        # OpenAI, GitHub PAT, Google API key, and generic long secrets
        "api_key": re.compile(
            r"\b("
            r"sk-[A-Za-z0-9]{20,}"  # OpenAI
            r"|ghp_[A-Za-z0-9]{36}"  # GitHub PAT
            r"|AIza[A-Za-z0-9\-_]{35}"  # Google API key
            r"|[A-Za-z0-9]{32,64}"  # Generic long token (≥32 hex/alphanum chars)
            r")\b"
        ),
        "ip_address": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "credential_context": re.compile(
            r"(?i)(?:password|passwd|secret|token|api[_\-]?key)\s*[:=]\s*\S+"
        ),
    }

    # Minimum risk level assigned to each PII type
    _TYPE_RISK: Dict[str, str] = {
        "ssn": "high",
        "credit_card": "high",
        "api_key": "high",
        "email": "medium",
        "phone": "medium",
        "ip_address": "low",
        "credential_context": "low",
    }

    def __init__(self) -> None:
        self.config = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> PIIScanResult:
        """Run regex-only scan on a block of text."""
        matches: Dict[str, List[str]] = {}

        for pii_type, pattern in self.REGEX_PATTERNS.items():
            found = pattern.findall(text)
            if found:
                # findall returns tuples for groups; flatten to strings
                flat: List[str] = []
                for item in found:
                    if isinstance(item, tuple):
                        flat.append("".join(item))
                    else:
                        flat.append(item)
                matches[pii_type] = flat

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
        for pii_type, pattern in self.REGEX_PATTERNS.items():
            placeholder = f"[REDACTED-{pii_type.upper().replace('_', '-')}]"
            text = pattern.sub(placeholder, text)
        return text

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
