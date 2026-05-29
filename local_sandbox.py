"""Local sandbox server compatible with SandboxFusion API format.
No Docker required - runs code directly via subprocess.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return "Local Sandbox is running"

@app.route("/run_code", methods=["POST"])
def run_code():
    data = request.json or {}
    code = data.get("code", "")
    language = data.get("language", "python")
    stdin_data = data.get("stdin", None)
    run_timeout = data.get("run_timeout", 60)
    compile_timeout = data.get("compile_timeout", 60)

    if language != "python":
        return jsonify({
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Error",
                "return_code": 1,
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "execution_time": 0
            }
        })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        temp_path = tmp_file.name

    try:
        timeout = max(run_timeout, compile_timeout)
        start = time.time()
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            input=stdin_data,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return jsonify({
            "status": "Success" if result.returncode == 0 else "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": elapsed
            }
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "TimeLimitExceeded",
                "return_code": -1,
                "stdout": "",
                "stderr": "Execution timed out",
                "execution_time": timeout
            }
        })
    except Exception as e:
        return jsonify({
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Error",
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0
            }
        })
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    host = os.getenv("LOCAL_SANDBOX_HOST", "0.0.0.0")
    port = int(os.getenv("LOCAL_SANDBOX_PORT", "8080"))
    print(f"Local sandbox server starting on {host}:{port}")
    app.run(host=host, port=port, threaded=True)
