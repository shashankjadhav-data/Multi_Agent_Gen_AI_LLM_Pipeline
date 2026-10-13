import contextlib
import io
import traceback
from typing import Any, Dict, Tuple

from utils.logger import logger

_BLOCKED = [
    "import subprocess", "import sys", "import os",
    "__import__", "os.system", "os.popen",
    "shutil.rmtree", "shutil.move",
]

class PythonRuntime:

    def execute(
        self,
        code: str,
        context: Dict[str, Any] = None,
    ) -> Tuple[str, str, bool]:
        for token in _BLOCKED:
            if token in code:
                msg = f"Blocked expression: '{token}'"
                logger.warning(f"[Runtime] {msg}")
                return "", msg, False

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        ns: Dict[str, Any] = {}
        if context:
            ns.update(context)

        try:
            compiled = compile(code, "<sandbox>", "exec")
            with contextlib.redirect_stdout(stdout_buf):
                with contextlib.redirect_stderr(stderr_buf):
                    exec(compiled, ns)
            return stdout_buf.getvalue(), stderr_buf.getvalue(), True
        except Exception:
            return stdout_buf.getvalue(), traceback.format_exc(), False

python_runtime = PythonRuntime()
