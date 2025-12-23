from typing import List

import requests

from .models import Conversation


class OfflineTagGenerator:
    """Generate tags using local Ollama LLM"""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate_tags(self, conversation: Conversation) -> List[str]:
        """
        Generate 3-5 relevant tags for a conversation
        Uses LLM if available, falls back to keyword extraction
        """

        if not self.is_available():
            print("⚠️  Ollama not running. Using keyword extraction fallback.")
            return self._fallback_tags(conversation)

        # Create focused prompt with conversation context
        first_msg = (
            conversation.messages[0].content[:400] if conversation.messages else ""
        )
        last_msg = (
            conversation.messages[-1].content[:400]
            if len(conversation.messages) > 1
            else ""
        )

        prompt = f"""You are a tag generator. Analyze this conversation and ONLY generate and output exactly 3-5 relevant tags for categorization.

        Title: {conversation.title}
        First message: {first_msg}
        Last message: {last_msg}

        CRITICAL RULES:
        - Output format: word1, word2, word3
        - Use commas to separate tags
        - No numbers, no hashtags, no bullets
        - Lowercase only
        - 3-5 tags maximum
        - Do NOT include any explanation, only the tags
        - Tags should be concise (1-3 words each), relevant to the content
        - Avoid overly generic tags like 'chat', 'conversation', 'general'
        - Prefer specific technical or topical tags

        Example correct output: python, export-data, tutorial, json-format

        Your answer should be only the tags comma-separated) without any additional text."""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for consistency
                        "num_predict": 60,  # Limit output length
                        "top_p": 0.9,
                    },
                },
                timeout=15,
            )

            if response.status_code == 200:
                tags_text = response.json()["response"].strip()

                # Parse and clean tags
                tags = self._parse_tags(tags_text)
                return tags[:5]  # Limit to 5 tags

        except requests.exceptions.Timeout:
            print("⚠️  Ollama request timed out. Using fallback.")
        except Exception as e:
            print(f"⚠️  Error generating tags: {e}. Using fallback.")

        return self._fallback_tags(conversation)

    def _parse_tags(self, tags_text: str) -> List[str]:
        """Parse and clean tag text from LLM response"""
        # Remove common prefixes the LLM might add
        tags_text = tags_text.replace("Your tags (comma-separated):", "").replace(
            "tags:", ""
        )
        tags_text = tags_text.strip()

        # Split by comma
        tags = [tag.strip().lower() for tag in tags_text.split(",")]

        # Filter out invalid tags
        valid_tags = []
        for tag in tags:
            # Remove quotes, periods, etc.
            tag = tag.strip(".\"'")

            # Only keep reasonable tags (1-25 chars, alphanumeric + hyphens)
            if 2 <= len(tag) <= 25 and all(c.isalnum() or c in ["-", "_"] for c in tag):
                valid_tags.append(tag)

        return valid_tags

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

        title_lower = conversation.title.lower()
        content_lower = (
            conversation.messages[0].content[:500].lower()
            if conversation.messages
            else ""
        )
        combined = f"{title_lower} {content_lower}"

        tags = []
        for tag, patterns in keywords.items():
            if any(pattern in combined for pattern in patterns):
                tags.append(tag)

        return tags[:5] if tags else ["general"]
