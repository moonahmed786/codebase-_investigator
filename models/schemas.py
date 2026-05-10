from pydantic import BaseModel
from typing import Optional, List, Literal


class CreateSessionRequest(BaseModel):
    github_url: str


class RepoInfo(BaseModel):
    owner: str
    repo: str
    branch: str
    url: str
    description: Optional[str] = None
    language: Optional[str] = None
    file_count: int = 0


class CreateSessionResponse(BaseModel):
    session_id: str
    repo_info: RepoInfo


class ChatRequest(BaseModel):
    session_id: str
    message: str


class CitationCheck(BaseModel):
    raw: str          # e.g. "auth/views.py:L45-L60"
    file: str
    start_line: int
    end_line: Optional[int] = None
    valid: bool
    reason: Optional[str] = None   # why invalid, if applicable
    snippet: Optional[str] = None  # actual lines if valid


class AuditFlag(BaseModel):
    severity: Literal["error", "warning", "info"]
    text: str


class AuditResult(BaseModel):
    verdict: Literal["trustworthy", "caution", "unreliable"]
    citation_checks: List[CitationCheck]
    flags: List[AuditFlag]
    contradictions: List[str]
    missing_context: List[str]
    summary: str


class ConversationTurn(BaseModel):
    turn: int
    question: str
    answer: str
    audit: AuditResult


class ChatResponse(BaseModel):
    turn: int
    answer: str
    audit: AuditResult


class ProgressEvent(BaseModel):
    type: Literal["tool_call", "tool_result", "status", "answer", "audit", "error"]
    data: dict
