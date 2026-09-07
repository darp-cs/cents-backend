from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


_RUNNER_SCRIPT = r'''
import asyncio
import inspect
import json
import traceback
import sys


def _emit(payload):
    print(json.dumps(payload, ensure_ascii=True))


def _main():
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")

    code = str(payload.get("code", ""))
    entrypoint = str(payload.get("entrypoint", "run")).strip() or "run"
    tool_input = payload.get("input", {})
    context = payload.get("context", {})

    namespace = {"__builtins__": __builtins__}

    try:
        exec(compile(code, "<tool-python>", "exec"), namespace)
        callable_object = namespace.get(entrypoint)
        if not callable(callable_object):
            raise RuntimeError(f"Entrypoint '{entrypoint}' is not defined or is not callable.")

        result = None
        try:
            result = callable_object(tool_input, context)
        except TypeError as exc:
            try:
                result = callable_object(tool_input)
            except TypeError:
                raise exc

        if inspect.isawaitable(result):
            result = asyncio.run(result)

        json.dumps(result)
        _emit({"ok": True, "output": result})
    except Exception as exc:
        _emit({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


if __name__ == "__main__":
    _main()
'''


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, *, traceback_text: str | None = None):
        super().__init__(message)
        self.traceback_text = traceback_text


@dataclass(frozen=True)
class ToolExecutionResult:
    output: Any


def compile_python_tool_code(code: str) -> None:
    compile(code, "<tool-python>", "exec")


def execute_python_tool_code(
    code: str,
    *,
    tool_input: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    entrypoint: str = "run",
    timeout_seconds: int = 20,
) -> ToolExecutionResult:
    input_payload = {
        "code": code,
        "entrypoint": entrypoint,
        "input": tool_input or {},
        "context": context or {},
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUNNER_SCRIPT],
            input=json.dumps(input_payload, ensure_ascii=True),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError("Tool code execution timed out.") from exc
    except OSError as exc:
        raise ToolExecutionError(f"Tool code execution failed to start: {exc}") from exc

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if not stdout:
        message = "Tool code execution produced no output."
        if stderr:
            message = f"{message} stderr: {stderr}"
        raise ToolExecutionError(message)

    last_line = stdout.splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("Tool code execution returned invalid JSON output.") from exc

    if not isinstance(payload, dict):
        raise ToolExecutionError("Tool code execution returned a malformed payload.")

    if payload.get("ok") is not True:
        error_text = str(payload.get("error", "Tool code execution failed."))
        traceback_text = str(payload.get("traceback", "")).strip() or None
        raise ToolExecutionError(error_text, traceback_text=traceback_text)

    return ToolExecutionResult(output=payload.get("output"))
