"""
CreditSage AI Advisor — Session Memory

Manages conversation history within a single Streamlit session.
Memory is stored in OpenAI message format and automatically trimmed
to avoid exceeding context window limits.
"""


class SessionMemory:
    """
    In-session conversation memory for the CreditSage AI Advisor.

    Stores messages in OpenAI format:
        [{"role": "user" | "assistant", "content": "..."}]

    Automatically trims to the last `max_history` messages to keep
    the context window manageable without losing recent context.
    """

    def __init__(self, max_history: int = 20):
        """
        Args:
            max_history: Maximum number of messages to retain.
                         20 = ~10 conversation turns, balances context vs. cost.
        """
        self.messages: list[dict] = []
        self.max_history = max_history

    def add_user(self, content: str) -> None:
        """Add a user message to memory."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Add an assistant message to memory."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> list[dict]:
        """Return a copy of the current conversation history."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all conversation history (e.g., on applicant change or reset)."""
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)

    def _trim(self) -> None:
        """Keep only the most recent max_history messages."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
