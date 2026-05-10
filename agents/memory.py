from typing import List
from models.schemas import ConversationTurn


class ConversationMemory:
    """
    Stores the full turn history and builds context windows for agents.
    Compresses old turns once the history grows large so neither agent
    blows its context window.
    """

    MAX_FULL_TURNS = 12       # keep this many turns verbatim
    COMPRESS_AFTER = 8        # summarise turns older than this

    def __init__(self):
        self._turns: List[ConversationTurn] = []

    def add_turn(self, turn: ConversationTurn) -> None:
        self._turns.append(turn)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def history_for_investigator(self) -> List[dict]:
        """
        Anthropic messages format for the investigation agent.
        Older turns are compressed; recent ones are kept verbatim.
        Only question + final answer (no internal tool use).
        """
        turns = self._turns
        messages: List[dict] = []

        if len(turns) > self.MAX_FULL_TURNS:
            old = turns[: -self.MAX_FULL_TURNS]
            recent = turns[-self.MAX_FULL_TURNS :]
            summary_lines = [
                f"Turn {t.turn}: Q: {t.question[:120]}  A: {t.answer[:300]}..."
                for t in old
            ]
            summary = "\n".join(summary_lines)
            messages.append({
                "role": "user",
                "content": f"[EARLIER CONVERSATION SUMMARY]\n{summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I have the earlier context in mind.",
            })
            turns = recent

        for t in turns:
            messages.append({"role": "user", "content": t.question})
            messages.append({"role": "assistant", "content": t.answer})

        return messages

    def history_for_auditor(self) -> str:
        """
        Plain-text summary of all prior turns for the audit prompt.
        """
        if not self._turns:
            return "No prior conversation turns."
        lines = []
        for t in self._turns:
            lines.append(f"--- Turn {t.turn} ---")
            lines.append(f"USER: {t.question}")
            lines.append(f"ASSISTANT: {t.answer[:800]}{'...' if len(t.answer) > 800 else ''}")
            lines.append("")
        return "\n".join(lines)

    def all_prior_claims(self) -> str:
        """
        Extracts a flat list of claims from all prior answers.
        Used by the auditor to surface contradictions.
        """
        if not self._turns:
            return "No prior claims."
        lines = []
        for t in self._turns:
            lines.append(f"Turn {t.turn} ({t.question[:80]!r}): {t.answer[:600]}")
        return "\n\n".join(lines)
