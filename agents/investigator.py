import json
from typing import AsyncIterator, List, Optional

import anthropic

from tools.github import GitHubClient

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 12

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the repository. Optionally limit to a line range. "
            "Always prefer reading specific line ranges for large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed, inclusive)"},
                "end_line": {"type": "integer", "description": "Last line to read (1-indexed, inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_codebase",
        "description": (
            "Search for a string or pattern across all text files. "
            "Returns matching files with surrounding context lines. "
            "Use file_pattern to narrow to specific file types, e.g. '*.py'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for (case-insensitive)"},
                "file_pattern": {"type": "string", "description": "Glob pattern to filter files, e.g. '*.py'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_directory",
        "description": "List the immediate contents of a directory. Use '' for the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (empty string = root)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_files",
        "description": "Find files by name or glob pattern anywhere in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*auth*', '*.test.ts'"},
            },
            "required": ["pattern"],
        },
    },
]


def _system_prompt(owner: str, repo: str, branch: str) -> str:
    return f"""\
You are an expert code investigator analysing the GitHub repository \
`{owner}/{repo}` (branch: `{branch}`).

TOOLS: Use them freely to read files, search for patterns, and list directories. \
Prefer targeted reads over full-file reads when files are large.

CITATION FORMAT — mandatory:
Every concrete claim about code **must** be backed by an inline citation in this \
exact format: `[path/to/file.py:L10]` (single line) or `[path/to/file.py:L10-L25]` \
(range). Citations go immediately after the statement they support, inside the \
sentence, like a footnote. Example:
  "Authentication is handled via JWT tokens set in the `login` view \
[auth/views.py:L45-L60]."

RULES:
1. Never state something about the code without a citation from a file you actually read.
2. Never guess line numbers — look them up with read_file or search_codebase first.
3. Keep answers focused: investigate only what the question asks, but go deep enough \
   to give a grounded, specific answer.
4. When the user references an earlier answer, check the conversation history and \
   either confirm, refine, or explicitly correct what you said before.
5. Flag risks, surprising choices, and dead code when you find them.
6. Opinions are welcome — mark them as such with "My take:" or similar.
"""


async def run_investigation(
    user_message: str,
    github: GitHubClient,
    conversation_history: List[dict],
    anthropic_client: anthropic.AsyncAnthropic,
    progress_cb=None,
) -> str:
    """
    Runs the investigation agent tool-use loop and returns the final answer text.
    progress_cb(event_type, data) is called for SSE progress events.
    """
    system = _system_prompt(github.owner, github.repo, github.branch)
    messages = [*conversation_history, {"role": "user", "content": user_message}]

    async def emit(event_type: str, data: dict):
        if progress_cb:
            await progress_cb(event_type, data)

    for _round in range(MAX_TOOL_ROUNDS):
        response = await anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_parts).strip()

        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_parts).strip()

        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            inp = block.input
            await emit("tool_call", {"tool": name, "input": inp})

            result = await _execute_tool(name, inp, github)

            await emit("tool_result", {
                "tool": name,
                "summary": _summarise_result(name, result),
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result if isinstance(result, str) else json.dumps(result),
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Fallback: extract any text from the last assistant turn
    last_text = [b.text for b in response.content if hasattr(b, "text")]
    return "\n".join(last_text).strip() or "Investigation reached tool limit without a final answer."


async def _execute_tool(name: str, inp: dict, github: GitHubClient) -> str:
    try:
        if name == "read_file":
            path = inp["path"]
            start = inp.get("start_line")
            end = inp.get("end_line")
            if start:
                content = await github.get_file_lines(path, start, end)
                if content is None:
                    return f"ERROR: File '{path}' not found or could not be fetched."
                return content
            else:
                content = await github.get_file(path)
                if content is None:
                    return f"ERROR: File '{path}' not found or could not be fetched."
                lines = content.splitlines()
                if len(lines) > 300:
                    preview = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:300]))
                    return preview + f"\n\n[File truncated — {len(lines)} total lines. Use start_line/end_line to read more.]"
                return "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))

        elif name == "search_codebase":
            results = await github.search_codebase(
                query=inp["query"],
                file_pattern=inp.get("file_pattern"),
            )
            if not results:
                return f"No matches found for '{inp['query']}'."
            parts = []
            for r in results:
                parts.append(f"FILE: {r['file']}")
                for m in r["matches"]:
                    parts.append(f"  Line {m['line']}:\n{m['snippet']}")
                parts.append("")
            return "\n".join(parts)

        elif name == "list_directory":
            items = await github.list_directory(inp.get("path", ""))
            if not items:
                return "Directory is empty or not found."
            lines = []
            for item in sorted(items, key=lambda x: (x["type"] == "file", x["name"])):
                icon = "📁" if item["type"] == "dir" else "📄"
                lines.append(f"{icon} {item['path']}")
            return "\n".join(lines)

        elif name == "find_files":
            paths = await github.find_files(inp["pattern"])
            if not paths:
                return f"No files matching '{inp['pattern']}' found."
            return "\n".join(paths[:60])

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {e}"


def _summarise_result(tool: str, result: str) -> str:
    lines = result.splitlines()
    count = len(lines)
    if tool == "search_codebase":
        files = sum(1 for l in lines if l.startswith("FILE:"))
        return f"Found matches in {files} file(s)"
    if tool == "list_directory":
        return f"Listed {count} item(s)"
    if tool == "find_files":
        return f"Found {count} file(s)"
    if tool == "read_file":
        return f"Read {count} line(s)"
    return f"{count} line(s) returned"
