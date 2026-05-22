from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class CodingTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_command: str = ""
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _jsonish_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _jsonish_list(decoded)
    return [str(value)]


def coding_task_from_row(row: Dict[str, Any], default_test_command: str = "") -> CodingTask:
    instance_id = str(row.get("instance_id") or row.get("id") or row.get("task_id") or "")
    if not instance_id:
        raise ValueError("Coding task row is missing instance_id/id/task_id.")

    problem_statement = str(row.get("problem_statement") or row.get("question") or row.get("prompt") or "")
    if not problem_statement:
        raise ValueError(f"Coding task {instance_id} is missing problem_statement/question/prompt.")

    test_command = str(row.get("test_command") or row.get("test_cmd") or row.get("command") or default_test_command or "")
    return CodingTask(
        instance_id=instance_id,
        repo=str(row.get("repo") or row.get("repository") or row.get("repo_path") or ""),
        base_commit=str(row.get("base_commit") or row.get("commit") or row.get("base_sha") or ""),
        problem_statement=problem_statement,
        test_command=test_command,
        fail_to_pass=_jsonish_list(row.get("FAIL_TO_PASS") or row.get("fail_to_pass")),
        pass_to_pass=_jsonish_list(row.get("PASS_TO_PASS") or row.get("pass_to_pass")),
        raw=dict(row),
    )


def truncate_output(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def run_bash(command: str, cwd: str, timeout_seconds: int, output_max_chars: int) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        stdout = truncate_output(completed.stdout, output_max_chars)
        stderr = truncate_output(completed.stderr, output_max_chars)
        return {
            "command": command,
            "cwd": cwd,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            "output": truncate_output(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}", output_max_chars),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = truncate_output(exc.stdout or "", output_max_chars)
        stderr = truncate_output(exc.stderr or "", output_max_chars)
        return {
            "command": command,
            "cwd": cwd,
            "exit_code": 124,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "output": truncate_output(f"Timed out after {timeout_seconds}s.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}", output_max_chars),
        }


class WorkspaceManager:
    def __init__(self, root: str, keep: bool = False):
        self.root = Path(root).resolve()
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]

    def _source_repo_path(self, task: CodingTask) -> Path:
        repo = Path(task.repo).expanduser()
        if repo.exists():
            return repo.resolve()
        candidate = self.root / "repos" / self.safe_name(task.repo)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Cannot find repo for {task.instance_id}. Set row repo/repo_path to a local path "
            f"or place it at {candidate}."
        )

    def create_root(self, task: CodingTask) -> str:
        path = self.root / self.safe_name(task.instance_id) / "0"
        if path.exists():
            shutil.rmtree(path)
        source = self._source_repo_path(task)
        subprocess.run(["git", "clone", "--quiet", str(source), str(path)], check=True)
        if task.base_commit:
            subprocess.run(["git", "checkout", "--quiet", task.base_commit], cwd=path, check=True)
        return str(path)

    def create_child(self, parent_workspace: str, child_tag: str) -> str:
        parent = Path(parent_workspace)
        child = parent.parent / self.safe_name(child_tag)
        if child.exists():
            shutil.rmtree(child)
        shutil.copytree(parent, child, symlinks=True)
        return str(child)

    @staticmethod
    def git_diff(workspace: str, output_max_chars: int = 200000) -> str:
        completed = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            return completed.stderr
        return truncate_output(completed.stdout, output_max_chars)
