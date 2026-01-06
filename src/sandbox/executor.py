"""
Sandboxed command executor for incident-aware-agents.

Used as the shell_executor backing ns.governed_act(kind="shell").

Safety goals:
- No network: --network none
- No privilege escalation: --security-opt no-new-privileges + cap-drop ALL
- Constrained resources: --cpus, --memory, --pids-limit, timeout
- Read-only root filesystem: --read-only + tmpfs for /tmp
- Workspace mount is the only writable area

NOTE:
This is still an experiment sandbox. For production, tighten further:
- seccomp/apparmor profiles
- explicit allowlist of binaries
- syscall filtering
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_IMAGE = "incident-sandbox"
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = "0.5"
DEFAULT_PIDS_LIMIT = "128"


@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    @property
    def output(self) -> str:
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        return "\n".join(parts)


class SandboxExecutor:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        pids_limit: str = DEFAULT_PIDS_LIMIT,
        workspace: Path | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self._workspace = workspace

    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="sandbox_"))
        return self._workspace

    def run(
        self,
        command: str,
        *,
        cwd: str = "/workspace",
        timeout: int | None = None,
        capture: bool = True,
    ) -> ExecutionResult:
        timeout = timeout or self.timeout
        docker_cmd = self._build_docker_command(command, cwd=cwd)

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                command=command,
                exit_code=proc.returncode,
                stdout=(proc.stdout or "").strip(),
                stderr=(proc.stderr or "").strip(),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(e),
            )

    def _build_docker_command(self, command: str, *, cwd: str) -> list[str]:
        # Force execution under /bin/sh -c to preserve your generated shell commands.
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            self.pids_limit,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{self.workspace}:/workspace:rw",
            "-w",
            cwd if cwd.startswith("/") else "/workspace",
            # Run as the non-root user created in your Dockerfile (named "sandbox")
            "--user",
            "sandbox",
            self.image,
            "/bin/sh",
            "-c",
            command,
        ]

    def cleanup(self) -> None:
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace)
        self._workspace = None

    def __enter__(self) -> "SandboxExecutor":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


class LocalExecutor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def run(
        self,
        command: str,
        *,
        timeout: int | None = None,
        capture: bool = True,
    ) -> ExecutionResult:
        timeout = timeout or self.timeout
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                command=command,
                exit_code=proc.returncode,
                stdout=(proc.stdout or "").strip(),
                stderr=(proc.stderr or "").strip(),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(command=command, exit_code=-1, stdout="", stderr="", timed_out=True)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(command=command, exit_code=-1, stdout="", stderr="", error=str(e))


ExecutorMode = Literal["sandbox", "local"]
_default_executor: SandboxExecutor | None = None


def create_executor(mode: ExecutorMode = "sandbox", **kwargs) -> SandboxExecutor | LocalExecutor:
    if mode == "sandbox":
        return SandboxExecutor(**kwargs)
    return LocalExecutor(**kwargs)


def execute_in_sandbox(
    command: str,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: int | None = None,
    cwd: str = "/workspace",
) -> str:
    """
    Convenience wrapper for ns.set(shell_executor=...).

    Returns combined stdout/stderr.
    """
    global _default_executor
    if _default_executor is None or _default_executor.image != image:
        _default_executor = SandboxExecutor(image=image)

    result = _default_executor.run(command, cwd=cwd, timeout=timeout)
    return result.output