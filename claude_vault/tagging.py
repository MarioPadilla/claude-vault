"""Tag generation for conversations using Ollama LLM"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from .models import Conversation


@dataclass
class TagResult:
    """Result of tag generation with optional hierarchy"""

    tags: List[str]  # Flat list for backward compatibility
    primary: List[str] = None  # Main topics (most relevant)
    secondary: List[str] = None  # Supporting topics
    tertiary: List[str] = None  # Minor/peripheral topics

    def __post_init__(self):
        if self.primary is None:
            self.primary = []
        if self.secondary is None:
            self.secondary = []
        if self.tertiary is None:
            self.tertiary = []

    def all_tags(self) -> List[str]:
        """Get all tags as a flat list"""
        if self.primary or self.secondary or self.tertiary:
            return self.primary + self.secondary + self.tertiary
        return self.tags


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

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate_tags(
        self,
        conversation: Conversation,
        mode: str = "quick",
        use_hierarchy: bool = False,
    ) -> TagResult:
        """
        Generate tags for a conversation

        Args:
            conversation: Conversation object to tag
            mode: 'quick' (title + first/last message) or 'full' (entire content)
            use_hierarchy: If True, generate primary/secondary/tertiary tags

        Returns:
            TagResult with tags (and optionally hierarchical tags)
        """
        if not self.is_available():
            print("! Ollama not running. Using keyword extraction fallback.")
            tags = self._fallback_tags(conversation)
            return TagResult(tags=tags)

        if use_hierarchy:
            return self._generate_hierarchical_tags(conversation, mode)
        else:
            tags = self._generate_flat_tags(conversation, mode)
            return TagResult(tags=tags)

    def _generate_flat_tags(self, conversation: Conversation, mode: str) -> List[str]:
        """Generate flat list of 3-5 tags"""
        content = self._get_content_for_analysis(conversation, mode)

        prompt = f"""You are a tag generator. Analyze this conversation and ONLY output exactly 3-5 relevant tags for categorization.

Title: {conversation.title}
Content: {content}

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

Your answer should be only the tags (comma-separated) without any additional text."""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 60,
                        "top_p": 0.9,
                    },
                },
                timeout=30,
            )

            if response.status_code == 200:
                tags_text = response.json()["response"].strip()
                tags = self._parse_tags(tags_text)
                return tags[:5]

        except requests.exceptions.Timeout:
            print("! Ollama request timed out. Using fallback.")
        except Exception as e:
            print(f"! Error generating tags: {e}. Using fallback.")

        return self._fallback_tags(conversation)

    def _generate_hierarchical_tags(
        self, conversation: Conversation, mode: str
    ) -> TagResult:
        """Generate hierarchical tags (primary, secondary, tertiary)"""
        content = self._get_content_for_analysis(conversation, mode)

        prompt = f"""You are a tag generator. Analyze this conversation and categorize it with hierarchical tags.

Title: {conversation.title}
Content: {content}

Generate tags in THREE categories:
1. PRIMARY (2-3 tags): Core topics - the main subjects this conversation is about
2. SECONDARY (2-3 tags): Supporting topics - important but not the main focus
3. TERTIARY (1-2 tags): Minor topics - peripherally mentioned or context

CRITICAL RULES:
- Use this EXACT format:
PRIMARY: tag1, tag2, tag3
SECONDARY: tag1, tag2, tag3
TERTIARY: tag1, tag2
- Lowercase only
- No hashtags, no bullets, no numbers
- Tags should be concise (1-3 words each)
- Avoid generic tags like 'chat', 'conversation', 'question'
- Each line must start with the category name followed by colon

Example output:
PRIMARY: python, web-scraping, data-extraction
SECONDARY: requests-library, html-parsing, error-handling
TERTIARY: automation, scripting"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,
                        "top_p": 0.9,
                    },
                },
                timeout=45,
            )

            if response.status_code == 200:
                response_text = response.json()["response"].strip()
                return self._parse_hierarchical_tags(response_text)

        except requests.exceptions.Timeout:
            print("! Ollama request timed out. Using fallback.")
        except Exception as e:
            print(f"! Error generating hierarchical tags: {e}. Using fallback.")

        # Fallback to flat tags
        tags = self._fallback_tags(conversation)
        return TagResult(
            tags=tags,
            primary=tags[:2],
            secondary=tags[2:4] if len(tags) > 2 else [],
            tertiary=tags[4:] if len(tags) > 4 else [],
        )

    def _get_content_for_analysis(self, conversation: Conversation, mode: str) -> str:
        """Get content to analyze based on mode"""
        if mode == "full":
            # Full content analysis - combine all messages
            all_content = []
            for msg in conversation.messages:
                role = "User" if msg.role == "human" else "Assistant"
                # Limit each message to prevent token overflow
                content = msg.content[:2000] if len(msg.content) > 2000 else msg.content
                all_content.append(f"{role}: {content}")

            # Join with limit to prevent massive prompts
            full_text = "\n\n".join(all_content)
            # Limit total content to ~8000 chars for reasonable processing
            if len(full_text) > 8000:
                # Take beginning and end
                full_text = full_text[:4000] + "\n\n[...content truncated...]\n\n" + full_text[-4000:]
            return full_text
        else:
            # Quick mode - title + first/last message snippets
            first_msg = (
                conversation.messages[0].content[:400] if conversation.messages else ""
            )
            last_msg = (
                conversation.messages[-1].content[:400]
                if len(conversation.messages) > 1
                else ""
            )
            return f"First message: {first_msg}\n\nLast message: {last_msg}"

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

            # Only keep reasonable tags (2-25 chars, alphanumeric + hyphens)
            if 2 <= len(tag) <= 25 and all(c.isalnum() or c in ["-", "_"] for c in tag):
                valid_tags.append(tag)

        return valid_tags

    def _parse_hierarchical_tags(self, response_text: str) -> TagResult:
        """Parse hierarchical tag response from LLM"""
        primary = []
        secondary = []
        tertiary = []

        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            line_lower = line.lower()

            if line_lower.startswith("primary:"):
                tags_part = line.split(":", 1)[1] if ":" in line else ""
                primary = self._parse_tags(tags_part)
            elif line_lower.startswith("secondary:"):
                tags_part = line.split(":", 1)[1] if ":" in line else ""
                secondary = self._parse_tags(tags_part)
            elif line_lower.startswith("tertiary:"):
                tags_part = line.split(":", 1)[1] if ":" in line else ""
                tertiary = self._parse_tags(tags_part)

        # Ensure we have at least some tags
        all_tags = primary + secondary + tertiary
        if not all_tags:
            # Try to parse as flat tags if hierarchical parsing failed
            all_tags = self._parse_tags(response_text)
            primary = all_tags[:3]
            secondary = all_tags[3:5] if len(all_tags) > 3 else []
            tertiary = all_tags[5:] if len(all_tags) > 5 else []

        return TagResult(
            tags=primary + secondary + tertiary,
            primary=primary[:3],
            secondary=secondary[:3],
            tertiary=tertiary[:2],
        )

    def _fallback_tags(self, conversation: Conversation) -> List[str]:
        """Simple keyword extraction as fallback when LLM unavailable"""
        keywords = {
            "python": ["python", "py", "django", "flask", "fastapi"],
            "javascript": ["javascript", "js", "react", "node", "npm", "typescript"],
            "api": ["api", "rest", "graphql", "endpoint", "http"],
            "debugging": ["debug", "error", "bug", "fix", "issue", "traceback"],
            "code": ["code", "coding", "programming", "development", "function"],
            "tutorial": ["tutorial", "learn", "guide", "how-to", "example"],
            "export": ["export", "download", "backup", "import"],
            "design": ["design", "ui", "ux", "interface", "layout"],
            "research": ["research", "study", "analysis", "paper"],
            "data": ["data", "database", "sql", "analytics", "csv", "json"],
            "web": ["web", "website", "html", "css", "frontend", "backend"],
            "machine-learning": ["ml", "machine learning", "ai", "model", "neural"],
            "testing": ["test", "testing", "qa", "unit test", "pytest"],
            "devops": ["docker", "kubernetes", "ci/cd", "deploy", "aws", "cloud"],
            "security": ["security", "auth", "authentication", "encryption"],
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
