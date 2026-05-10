import json
import os
import uuid
from typing import Dict

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from agents.auditor import run_audit
from agents.investigator import run_investigation
from agents.memory import ConversationMemory
from models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationTurn,
    CreateSessionRequest,
    CreateSessionResponse,
    RepoInfo,
)
from tools.citation_checker import validate_citations
from tools.github import GitHubClient

load_dotenv()

app = FastAPI(title="Codebase Investigator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────────────────

class Session:
    def __init__(self, github: GitHubClient, repo_info: RepoInfo):
        self.github = github
        self.repo_info = repo_info
        self.memory = ConversationMemory()

_sessions: Dict[str, Session] = {}

def _anthropic_client() -> anthropic.AsyncAnthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.AsyncAnthropic(api_key=key)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/api/session", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    token = os.getenv("GITHUB_TOKEN")
    try:
        github = GitHubClient(req.github_url, token=token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        meta = await github.fetch_repo_meta()
        tree = await github.get_tree()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not access repository: {e}")

    repo_info = RepoInfo(
        owner=github.owner,
        repo=github.repo,
        branch=github.branch,
        url=req.github_url,
        description=meta.get("description"),
        language=meta.get("language"),
        file_count=len(tree),
    )

    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(github=github, repo_info=repo_info)

    return CreateSessionResponse(session_id=session_id, repo_info=repo_info)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    client = _anthropic_client()

    async def event_generator():
        events = []

        async def progress_cb(event_type: str, data: dict):
            payload = json.dumps({"type": event_type, **data})
            events.append(("progress", payload))

        # ── Phase 1: Investigation ────────────────────────────────────────
        yield {"event": "status", "data": json.dumps({"message": "Investigating…"})}

        # Drain any queued progress events lazily via callback
        queued: list = []

        async def collecting_cb(event_type: str, data: dict):
            queued.append((event_type, data))

        answer = await run_investigation(
            user_message=req.message,
            github=session.github,
            conversation_history=session.memory.history_for_investigator(),
            anthropic_client=client,
            progress_cb=collecting_cb,
        )

        # Emit tool-call events that were collected
        for event_type, data in queued:
            yield {"event": event_type, "data": json.dumps(data)}

        yield {"event": "answer", "data": json.dumps({"text": answer})}

        # ── Phase 2: Citation validation (programmatic) ──────────────────
        yield {"event": "status", "data": json.dumps({"message": "Validating citations…"})}
        citation_checks = await validate_citations(answer, session.github)
        yield {
            "event": "citations",
            "data": json.dumps([c.model_dump() for c in citation_checks]),
        }

        # ── Phase 3: Independent audit ───────────────────────────────────
        yield {"event": "status", "data": json.dumps({"message": "Running independent audit…"})}
        audit = await run_audit(
            answer=answer,
            citation_checks=citation_checks,
            prior_conversation=session.memory.history_for_auditor(),
            anthropic_client=client,
        )
        yield {"event": "audit", "data": json.dumps(audit.model_dump())}

        # ── Persist turn ─────────────────────────────────────────────────
        turn_number = session.memory.turn_count + 1
        session.memory.add_turn(ConversationTurn(
            turn=turn_number,
            question=req.message,
            answer=answer,
            audit=audit,
        ))

        yield {
            "event": "complete",
            "data": json.dumps({"turn": turn_number}),
        }

    return EventSourceResponse(event_generator())


@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    turns = session.memory._turns
    return {"turns": [t.model_dump() for t in turns]}


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Frontend not found</h1><p>Place index.html in static/</p>"


app.mount("/static", StaticFiles(directory="static"), name="static")
