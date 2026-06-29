import re
from typing import List

import requests

from .config import load_config
from .models import Conversation


class OfflineTagGenerator:
    """Generate tags using local Ollama LLM"""

    def __init__(self):
        self.config = load_config()
        self.ollama_url = self.config.ollama.url

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            # Parse base URL
            base_url = self.ollama_url.split("/api")[0]
            response = requests.get(base_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate_metadata(self, conversation: Conversation) -> dict:
        """
        Generate relevant tags and a brief summary for a conversation
        Uses LLM if available, falls back to keyword extraction
        """

        if not self.is_available():
            print("⚠️  Ollama not running. Using keyword extraction fallback.")
            return self._fallback_metadata(conversation)

        # Create focused prompt with conversation context
        first_msg = (
            conversation.messages[0].content[:400] if conversation.messages else ""
        )
        last_msg = (
            conversation.messages[-1].content[:400]
            if len(conversation.messages) > 1
            else ""
        )

        prompt = f"""You are a conversation analyzer. Analyze this conversation and output a JSON object with tags and a summary.

        Title: {conversation.title}
        First message: {first_msg}
        Last message: {last_msg}

        CRITICAL RULES:
        - Output MUST be valid JSON
        - Format: {{"tags": ["tag1", "tag2"], "summary": "One sentence summary"}}
        - Tags: 3-5 lowercase tags, no spaces, specific to content
        - Summary: 1-2 sentence summary of the main topic or insight
        - No markdown formatting (no ```json code blocks)

        Example output:
        {{"tags": ["python", "api", "error-handling"], "summary": "The user is debugging a ConnectionError in their Python API client and learning about retry logic."}}"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.config.ollama.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",  # Force JSON mode if model supports it
                    "options": {
                        "temperature": self.config.ollama.temperature,
                        "num_predict": 150,
                        "top_p": 0.9,
                    },
                },
                timeout=self.config.ollama.timeout,
            )

            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get("response", "").strip()

                # Check if it's already a dict (some Ollama versions/wrappers might parse it)
                if isinstance(content, dict):
                    return self._validate_metadata(content)

                # Parse JSON string
                import json

                try:
                    data = json.loads(content)
                    return self._validate_metadata(data)
                except json.JSONDecodeError:
                    # Fallback if valid JSON wasn't returned
                    tags = self._parse_tags(content)
                    return {"tags": tags[:5], "summary": None}

        except requests.exceptions.Timeout:
            print("⚠️  Ollama request timed out. Using fallback.")
        except Exception as e:
            print(f"⚠️  Error generating metadata: {e}. Using fallback.")

        return self._fallback_metadata(conversation)

    def _validate_metadata(self, data: dict) -> dict:
        """Validate and clean metadata"""
        tags = data.get("tags", [])
        summary = data.get("summary")

        # Clean tags
        valid_tags = []
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str):
                    tag = tag.strip().lower()
                    # Only keep reasonable tags
                    if 2 <= len(tag) <= 30 and all(
                        c.isalnum() or c in ["-", "_"] for c in tag
                    ):
                        valid_tags.append(tag)

        return {"tags": valid_tags[:5], "summary": str(summary) if summary else None}

    def _parse_tags(self, tags_text: str) -> List[str]:
        """Legacy helper for parsing simple tag lists"""
        # Remove common prefixes
        tags_text = tags_text.replace("Your tags (comma-separated):", "").replace(
            "tags:", ""
        )
        tags_text = (
            tags_text.replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
        )
        tags_text = tags_text.strip()

        # Split by comma
        tags = [tag.strip().lower() for tag in tags_text.split(",")]

        # Filter out invalid tags
        valid_tags = []
        for tag in tags:
            tag = tag.strip(".\"'")
            if 2 <= len(tag) <= 25 and all(c.isalnum() or c in ["-", "_"] for c in tag):
                valid_tags.append(tag)

        return valid_tags

    def _fallback_metadata(self, conversation: Conversation) -> dict:
        """Fallback when LLM unavailable"""
        return {"tags": self._fallback_tags(conversation), "summary": None}

    def _fallback_tags(self, conversation: Conversation) -> List[str]:
        """Simple keyword extraction as fallback when LLM unavailable"""
        keywords = {
            "python": ["python", "py", "django", "flask"],
            "javascript": ["javascript", "js", "react", "node", "npm"],
            "api": ["api", "rest", "graphql", "endpoint"],
            "debugging": ["debug", "error", "bug", "fix", "issue"],
            "code": ["code", "coding", "programming", "development"],
            "tutorial": ["tutorial", "learn", "guide", "how-to"],
            "export": ["export", "download", "backup"],
            "design": ["design", "ui", "ux", "interface"],
            "research": ["research", "study", "analysis"],
            "data": ["data", "database", "sql", "analytics"],
            "web": ["web", "website", "html", "css"],
            "machine-learning": ["ml", "machine learning", "ai", "model"],
            "testing": ["test", "testing", "qa", "unit test"],
        }

        # Merge with custom keywords from config
        if self.config.custom_keywords:
            keywords.update(self.config.custom_keywords)

        title_lower = conversation.title.lower()
        content_lower = (
            conversation.messages[0].content[:500].lower()
            if conversation.messages
            else ""
        )
        combined = f"{title_lower} {content_lower}"

        tags = []
        for tag, patterns in keywords.items():
            if any(
                re.search(r"\b" + re.escape(pattern) + r"\b", combined)
                for pattern in patterns
            ):
                tags.append(tag)

        return tags[:5] if tags else ["general"]
