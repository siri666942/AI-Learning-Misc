#!/usr/bin/env python
"""Track hand-written code lines for the karpathy folder.

Counting rules:
- .py files count as source code.
- .ipynb files count only code-cell source lines.
- Markdown, notebook outputs, metadata, data files, images, PDFs, and models are ignored.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path


CODE_SUFFIXES = {".py", ".ipynb"}
DEFAULT_EXCLUDES = {
    "karpathy/scripts/code_line_stats.py",
}


@dataclass(frozen=True)
class Change:
    added: int
    deleted: int
    files: int

    @property
    def changed(self) -> int:
        return self.added + self.deleted

    @property
    def net(self) -> int:
        return self.added - self.deleted


def run_git(repo: Path, args: list[str], *, binary: bool = False) -> bytes | str:
    cmd = ["git", "-C", str(repo), *args]
    if binary:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")


def git_root(start: Path) -> Path:
    root = run_git(start, ["rev-parse", "--show-toplevel"])
    return Path(str(root).strip())


def repo_relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def is_code_path(path: str, excludes: set[str]) -> bool:
    return Path(path).suffix in CODE_SUFFIXES and path not in excludes


def blob_at(repo: Path, commit: str | None, path: str) -> bytes | None:
    if not commit:
        return None
    try:
        return run_git(repo, ["show", f"{commit}:{path}"], binary=True)  # type: ignore[return-value]
    except subprocess.CalledProcessError:
        return None


def worktree_blob(repo: Path, path: str) -> bytes | None:
    full_path = repo / path
    if not full_path.exists() or not full_path.is_file():
        return None
    return full_path.read_bytes()


def code_lines(path: str, data: bytes | None) -> list[str]:
    if data is None:
        return []

    text = data.decode("utf-8", errors="replace")
    suffix = Path(path).suffix
    if suffix == ".py":
        return text.splitlines()

    if suffix == ".ipynb":
        try:
            notebook = json.loads(text)
        except json.JSONDecodeError:
            return []

        lines: list[str] = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, str):
                cell_lines = source.splitlines()
            else:
                cell_lines = "".join(source).splitlines()
            lines.extend(cell_lines)
            lines.append("")
        return lines

    return []


def diff_counts(old_lines: list[str], new_lines: list[str]) -> tuple[int, int]:
    added = 0
    deleted = 0
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deleted += i2 - i1
        if tag in {"replace", "insert"}:
            added += j2 - j1
    return added, deleted


def parent_commit(repo: Path, commit: str) -> str | None:
    row = run_git(repo, ["rev-list", "--parents", "-n", "1", commit]).split()
    return row[1] if len(row) > 1 else None


def changed_paths_for_commit(repo: Path, commit: str, scope: str, excludes: set[str]) -> list[str]:
    output = run_git(
        repo,
        ["diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit, "--", scope],
    )
    return [line for line in output.splitlines() if is_code_path(line, excludes)]


def commit_change(repo: Path, commit: str, scope: str, excludes: set[str]) -> Change:
    parent = parent_commit(repo, commit)
    added = 0
    deleted = 0
    files = 0

    for path in changed_paths_for_commit(repo, commit, scope, excludes):
        old = code_lines(path, blob_at(repo, parent, path))
        new = code_lines(path, blob_at(repo, commit, path))
        path_added, path_deleted = diff_counts(old, new)
        if path_added or path_deleted:
            added += path_added
            deleted += path_deleted
            files += 1

    return Change(added=added, deleted=deleted, files=files)


def commits(repo: Path, scope: str) -> list[tuple[str, str]]:
    output = run_git(
        repo,
        ["log", "--reverse", "--date=short", "--pretty=format:%H%x09%ad", "--", scope],
    )
    rows = []
    for line in output.splitlines():
        commit, date = line.split("\t")
        rows.append((commit, date))
    return rows


def changed_worktree_paths(repo: Path, scope: str, excludes: set[str]) -> list[str]:
    tracked = run_git(repo, ["diff", "--name-only", "--", scope]).splitlines()
    untracked = run_git(repo, ["ls-files", "--others", "--exclude-standard", "--", scope]).splitlines()
    return [line for line in [*tracked, *untracked] if is_code_path(line, excludes)]


def worktree_change(repo: Path, scope: str, excludes: set[str]) -> Change:
    added = 0
    deleted = 0
    files = 0
    for path in changed_worktree_paths(repo, scope, excludes):
        old = code_lines(path, blob_at(repo, "HEAD", path))
        new = code_lines(path, worktree_blob(repo, path))
        path_added, path_deleted = diff_counts(old, new)
        if path_added or path_deleted:
            added += path_added
            deleted += path_deleted
            files += 1
    return Change(added=added, deleted=deleted, files=files)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS code_line_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            source TEXT NOT NULL,
            source_key TEXT NOT NULL,
            added INTEGER NOT NULL,
            deleted INTEGER NOT NULL,
            changed INTEGER NOT NULL,
            net INTEGER NOT NULL,
            files INTEGER NOT NULL,
            scope TEXT NOT NULL,
            note TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(source, source_key)
        );

        CREATE VIEW IF NOT EXISTS daily_code_lines AS
        SELECT
            event_date,
            SUM(added) AS added,
            SUM(deleted) AS deleted,
            SUM(net) AS net,
            SUM(changed) AS changed,
            SUM(files) AS files
        FROM code_line_events
        GROUP BY event_date
        ORDER BY event_date;
        """
    )
    conn.commit()


def upsert_event(
    conn: sqlite3.Connection,
    *,
    event_date: str,
    source: str,
    source_key: str,
    change: Change,
    scope: str,
    note: str,
) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO code_line_events
            (event_date, source, source_key, added, deleted, changed, net, files, scope, note, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, source_key) DO UPDATE SET
            event_date = excluded.event_date,
            added = excluded.added,
            deleted = excluded.deleted,
            changed = excluded.changed,
            net = excluded.net,
            files = excluded.files,
            scope = excluded.scope,
            note = excluded.note,
            updated_at = excluded.updated_at
        """,
        (
            event_date,
            source,
            source_key,
            change.added,
            change.deleted,
            change.changed,
            change.net,
            change.files,
            scope,
            note,
            now,
        ),
    )


def refresh_history(conn: sqlite3.Connection, repo: Path, scope: str, excludes: set[str]) -> None:
    for commit, date in commits(repo, scope):
        change = commit_change(repo, commit, scope, excludes)
        if not change.changed:
            continue
        upsert_event(
            conn,
            event_date=date,
            source="git",
            source_key=commit,
            change=change,
            scope=scope,
            note="committed code only",
        )
    conn.commit()


def refresh_worktree(conn: sqlite3.Connection, repo: Path, scope: str, event_date: str, excludes: set[str]) -> None:
    change = worktree_change(repo, scope, excludes)
    if change.changed:
        upsert_event(
            conn,
            event_date=event_date,
            source="working-tree",
            source_key=f"{scope}:working-tree",
            change=change,
            scope=scope,
            note="current uncommitted code diff against HEAD",
        )
    else:
        conn.execute(
            "DELETE FROM code_line_events WHERE source = ? AND source_key = ?",
            ("working-tree", f"{scope}:working-tree"),
        )
    conn.commit()


def print_daily(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT event_date, added, deleted, net, changed, files
        FROM daily_code_lines
        ORDER BY event_date
        """
    ).fetchall()
    print("date        added  deleted  net   changed  files")
    for row in rows:
        print(f"{row[0]}  {row[1]:>5}  {row[2]:>7}  {row[3]:>4}  {row[4]:>7}  {row[5]:>5}")


def print_events(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT event_date, source, source_key, added, deleted, net, changed, files
        FROM code_line_events
        ORDER BY event_date, source, source_key
        """
    ).fetchall()
    print("date        source        added  deleted  net   changed  files  key")
    for row in rows:
        key = row[2][:12] if row[1] == "git" else row[2]
        print(f"{row[0]}  {row[1]:<12}  {row[3]:>5}  {row[4]:>7}  {row[5]:>4}  {row[6]:>7}  {row[7]:>5}  {key}")


def default_paths() -> tuple[Path, str, Path]:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    repo = git_root(project_dir)
    scope = repo_relative(project_dir, repo)
    db_path = project_dir / "code_lines.sqlite"
    return repo, scope, db_path


def main() -> None:
    repo, scope, default_db = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["init", "update", "show", "events"],
        help="init/update the database or print stored stats",
    )
    parser.add_argument("--db", type=Path, default=default_db, help="SQLite database path")
    parser.add_argument("--scope", default=scope, help="Git pathspec to count")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Repo-relative file path to exclude; may be passed multiple times",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date used for the working-tree event; default is today",
    )
    args = parser.parse_args()

    conn = connect(args.db)
    init_db(conn)
    excludes = set(DEFAULT_EXCLUDES)
    excludes.update(path.replace(os.sep, "/") for path in args.exclude)

    if args.command in {"init", "update"}:
        refresh_history(conn, repo, args.scope, excludes)
        refresh_worktree(conn, repo, args.scope, args.date, excludes)
        print_daily(conn)
    elif args.command == "show":
        print_daily(conn)
    elif args.command == "events":
        print_events(conn)


if __name__ == "__main__":
    main()
