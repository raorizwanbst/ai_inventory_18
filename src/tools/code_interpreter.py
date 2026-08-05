from langchain_core.tools import tool
import subprocess
import tempfile
from pathlib import Path

@tool
def run_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess and return stdout/stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code)
        try:
            result = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
            return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nreturncode: {result.returncode}"
        except subprocess.TimeoutExpired:
            return "Error: execution timed out"
