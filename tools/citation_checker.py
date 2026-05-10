import re
from typing import List

from models.schemas import CitationCheck
from tools.github import GitHubClient

# Matches [path/to/file.py:L10] or [path/to/file.py:L10-L25]
CITATION_RE = re.compile(
    r"\[([^\]]+?):L(\d+)(?:-L(\d+))?\]"
)


def extract_citations(text: str) -> List[tuple]:
    """Return list of (raw_str, file_path, start_line, end_line|None)."""
    results = []
    for m in CITATION_RE.finditer(text):
        raw = m.group(0)
        path = m.group(1).strip()
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else None
        results.append((raw, path, start, end))
    return results


async def validate_citations(text: str, github: GitHubClient) -> List[CitationCheck]:
    citations_raw = extract_citations(text)
    if not citations_raw:
        return []

    checks: List[CitationCheck] = []
    for raw, path, start, end in citations_raw:
        # Check file exists in tree
        if not github.file_exists(path):
            checks.append(CitationCheck(
                raw=raw, file=path, start_line=start, end_line=end,
                valid=False, reason="File not found in repository"
            ))
            continue

        # Fetch file and validate line range
        content = await github.get_file(path)
        if content is None:
            checks.append(CitationCheck(
                raw=raw, file=path, start_line=start, end_line=end,
                valid=False, reason="Could not fetch file content"
            ))
            continue

        total_lines = len(content.splitlines())
        if start > total_lines:
            checks.append(CitationCheck(
                raw=raw, file=path, start_line=start, end_line=end,
                valid=False,
                reason=f"Line {start} out of range — file only has {total_lines} lines"
            ))
            continue

        if end and end > total_lines:
            checks.append(CitationCheck(
                raw=raw, file=path, start_line=start, end_line=end,
                valid=False,
                reason=f"End line {end} out of range — file only has {total_lines} lines"
            ))
            continue

        snippet = await github.get_file_lines(path, start, end)
        checks.append(CitationCheck(
            raw=raw, file=path, start_line=start, end_line=end,
            valid=True, snippet=snippet
        ))

    return checks
