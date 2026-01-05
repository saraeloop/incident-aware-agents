"""
Sandboxed command executor for incident-aware-agents.

Provides safe execution of shell commands in an isolated Docker container.
Used by the experiment to run tasks without risking the host system.

Safety guarantees:
    - No network access (--network none)
    - Read-only filesystem except /workspace
    - Resource limits (CPU, memory, timeout)
    - No privilege escalation

Usage:
    from sandbox.executor import SandboxExecutor

    executor = SandboxExecutor()
    result = executor.run("ls -la /workspace")
    print(result.stdout)
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_IMAGE = "incident-aware-sandbox:latest"
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = "0.5"


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of running a command in the sandbox."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        return "\n".join(parts)


# -----------------------------------------------------------------------------
# Sandbox Executor
# -----------------------------------------------------------------------------

class SandboxExecutor:
    """
    Executes commands in an isolated Docker container.

    The sandbox provides:
    - Isolation: No access to host filesystem or network
    - Safety: Commands can't escape or damage host
    - Reproducibility: Clean environment for each run

    Example:
        >>> executor = SandboxExecutor()
        >>> result = executor.run("echo hello")
        >>> result.stdout
        'hello'
        >>> result.success
        True
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        workspace: Path | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.memory = memory
        self.cpus = cpus
        self._workspace = workspace

    @property
    def workspace(self) -> Path:
        """Get or create workspace directory."""
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="sandbox_"))
        return self._workspace

    def run(
        self,
        command: str,
        timeout: int | None = None,
        capture: bool = True,
    ) -> ExecutionResult:
        """
        Run a command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Override default timeout (seconds)
            capture: Whether to capture stdout/stderr

        Returns:
            ExecutionResult with exit code, output, and status
        """
        timeout = timeout or self.timeout

        docker_cmd = self._build_docker_command(command)

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout.strip() if result.stdout else "",
                stderr=result.stderr.strip() if result.stderr else "",
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
        except Exception as e:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(e),
            )

    def _build_docker_command(self, command: str) -> list[str]:
        """Build the docker run command with security options."""
        return [
            "docker", "run",
            "--rm",                           # Remove container after exit
            "--network", "none",              # No network access
            "--read-only",                    # Read-only root filesystem
            "--memory", self.memory,          # Memory limit
            "--cpus", self.cpus,              # CPU limit
            "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "-v", f"{self.workspace}:/workspace:rw",
            "-w", "/workspace",
            self.image,
            "/bin/sh", "-c", command,
        ]

    def cleanup(self) -> None:
        """Remove workspace directory."""
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace)
            self._workspace = None

    def __enter__(self) -> "SandboxExecutor":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


# -----------------------------------------------------------------------------
# Local Executor (for testing without Docker)
# -----------------------------------------------------------------------------

class LocalExecutor:
    """
    Runs commands locally without sandboxing.

    Use only for development/testing. Not safe for untrusted commands.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def run(
        self,
        command: str,
        timeout: int | None = None,
        capture: bool = True,
    ) -> ExecutionResult:
        timeout = timeout or self.timeout

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout.strip() if result.stdout else "",
                stderr=result.stderr.strip() if result.stderr else "",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(e),
            )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

ExecutorMode = Literal["sandbox", "local", "dry-run"]
_default_executor: SandboxExecutor | None = None


def create_executor(
    mode: ExecutorMode = "sandbox",
    **kwargs,
) -> SandboxExecutor | LocalExecutor:
    """
    Create an executor based on mode.

    Args:
        mode: "sandbox" (Docker), "local" (no isolation), or "dry-run" (no-op)

    Returns:
        Configured executor instance
    """
    if mode == "sandbox":
        return SandboxExecutor(**kwargs)
    if mode == "local":
        return LocalExecutor(**kwargs)

    raise ValueError(f"Unknown mode: {mode}")


def execute_in_sandbox(
    command: str,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: int | None = None,
) -> str:
    """
    Convenience wrapper used by the agent.

    Delegates to SandboxExecutor and returns combined output.
    """
    global _default_executor
    if _default_executor is None or _default_executor.image != image:
        _default_executor = SandboxExecutor(image=image)

    result = _default_executor.run(command, timeout=timeout)
    return result.output


# -----------------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Executor Test")
    print("=" * 50)

    # Test local executor
    print("\nLocal executor:")
    local = LocalExecutor()
    result = local.run("echo 'hello world'")
    print(f"  Command: {result.command}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Output: {result.stdout}")
    print(f"  Success: {result.success}")

    # Test timeout
    print("\nTimeout test:")
    result = local.run("sleep 5", timeout=1)
    print(f"  Timed out: {result.timed_out}")

    print("\nTest complete.")
