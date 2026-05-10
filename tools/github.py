import fnmatch
import os
import re
from typing import Dict, List, Optional, Tuple

import httpx


TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".md", ".yaml", ".yml",
    ".json", ".toml", ".sh", ".bash", ".sql", ".html", ".css", ".scss",
    ".vue", ".svelte", ".kt", ".swift", ".dart", ".ex", ".exs", ".clj",
    ".hs", ".ml", ".tf", ".hcl", ".env.example", ".txt", ".cfg", ".ini",
    ".xml", ".graphql", ".proto", ".dockerfile", ".gitignore",
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "vendor", "venv", ".venv", "env", ".env",
    "coverage", ".coverage", "htmlcov", "eggs", ".eggs",
}


class GitHubClient:
    def __init__(self, github_url: str, token: Optional[str] = None):
        self.owner, self.repo, self.branch = self._parse_url(github_url)
        self.token = token
        self._tree: Optional[List[dict]] = None
        self._file_cache: Dict[str, Optional[str]] = {}
        self._repo_meta: Optional[dict] = None
        self._headers = {"Accept": "application/vnd.github+json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _parse_url(self, url: str) -> Tuple[str, str, str]:
        url = url.rstrip("/")
        # https://github.com/owner/repo or with /tree/branch or /blob/branch/...
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/([^/]+))?", url)
        if not m:
            raise ValueError(f"Cannot parse GitHub URL: {url}")
        owner, repo, branch = m.group(1), m.group(2), m.group(3) or "HEAD"
        repo = repo.removesuffix(".git")
        return owner, repo, branch

    @property
    def base_api(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}"

    async def fetch_repo_meta(self) -> dict:
        if self._repo_meta:
            return self._repo_meta
        async with httpx.AsyncClient(headers=self._headers, timeout=15, follow_redirects=True) as client:
            r = await client.get(self.base_api)
            r.raise_for_status()
            self._repo_meta = r.json()
            # Resolve default branch
            if self.branch == "HEAD":
                self.branch = self._repo_meta.get("default_branch", "main")
            return self._repo_meta

    async def get_tree(self) -> List[dict]:
        if self._tree is not None:
            return self._tree
        await self.fetch_repo_meta()
        async with httpx.AsyncClient(headers=self._headers, timeout=30, follow_redirects=True) as client:
            url = f"{self.base_api}/git/trees/{self.branch}?recursive=1"
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        raw_tree = data.get("tree", [])
        # Filter to text blobs, skip ignored dirs
        self._tree = [
            item for item in raw_tree
            if item.get("type") == "blob"
            and self._is_text_file(item["path"])
            and not self._in_skip_dir(item["path"])
        ]
        return self._tree

    def _is_text_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        basename = os.path.basename(path).lower()
        if ext in TEXT_EXTENSIONS:
            return True
        # extensionless files like Dockerfile, Makefile
        if basename in {"dockerfile", "makefile", "procfile", "rakefile", "gemfile", "brewfile"}:
            return True
        return False

    def _in_skip_dir(self, path: str) -> bool:
        parts = path.split("/")
        return any(p in SKIP_DIRS for p in parts[:-1])

    async def get_file(self, path: str) -> Optional[str]:
        if path in self._file_cache:
            return self._file_cache[path]
        url = f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}/{path}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code == 404:
                    self._file_cache[path] = None
                    return None
                r.raise_for_status()
                content = r.text
        except Exception:
            self._file_cache[path] = None
            return None
        self._file_cache[path] = content
        return content

    async def get_file_lines(self, path: str, start: int, end: Optional[int] = None) -> Optional[str]:
        content = await self.get_file(path)
        if content is None:
            return None
        lines = content.splitlines()
        total = len(lines)
        s = max(0, start - 1)
        e = min(total, (end or start))
        return "\n".join(
            f"{i+1}: {line}" for i, line in enumerate(lines[s:e], start=s)
        )

    async def list_directory(self, path: str = "") -> List[dict]:
        tree = await self.get_tree()
        path = path.strip("/")
        results = []
        seen_dirs = set()
        for item in tree:
            p = item["path"]
            if path:
                if not p.startswith(path + "/"):
                    continue
                relative = p[len(path) + 1:]
            else:
                relative = p
            parts = relative.split("/")
            if len(parts) == 1:
                results.append({"type": "file", "name": parts[0], "path": p})
            else:
                d = parts[0]
                full_dir = (path + "/" + d) if path else d
                if full_dir not in seen_dirs:
                    seen_dirs.add(full_dir)
                    results.append({"type": "dir", "name": d, "path": full_dir})
        return results

    async def find_files(self, pattern: str) -> List[str]:
        tree = await self.get_tree()
        pat_lower = pattern.lower()
        results = []
        for item in tree:
            p = item["path"]
            basename = os.path.basename(p)
            if fnmatch.fnmatch(p.lower(), pat_lower) or fnmatch.fnmatch(basename.lower(), pat_lower):
                results.append(p)
        return results

    async def search_codebase(
        self,
        query: str,
        file_pattern: Optional[str] = None,
        max_files: int = 25,
        context_lines: int = 3,
    ) -> List[dict]:
        tree = await self.get_tree()
        candidates = []
        for item in tree:
            p = item["path"]
            if file_pattern:
                basename = os.path.basename(p)
                if not fnmatch.fnmatch(p, file_pattern) and not fnmatch.fnmatch(basename, file_pattern):
                    continue
            candidates.append(p)

        # Prioritize files whose path contains query terms
        query_parts = query.lower().split()
        def score(path: str) -> int:
            pl = path.lower()
            return sum(1 for q in query_parts if q in pl)
        candidates.sort(key=score, reverse=True)

        results = []
        searched = 0
        for path in candidates:
            if searched >= max_files:
                break
            content = await self.get_file(path)
            if not content:
                continue
            searched += 1
            lines = content.splitlines()
            matches = []
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    snippet = "\n".join(
                        f"{'>' if j == i else ' '} {j+1}: {lines[j]}"
                        for j in range(start, end)
                    )
                    matches.append({"line": i + 1, "snippet": snippet})
            if matches:
                results.append({"file": path, "matches": matches[:8]})

        return results

    def file_exists(self, path: str) -> bool:
        if self._tree is None:
            return False
        return any(item["path"] == path for item in self._tree)

    def file_line_count(self, path: str) -> int:
        content = self._file_cache.get(path)
        if content is None:
            return 0
        return len(content.splitlines())
