"""Run the website and optional loopback GPU service in one Space."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from restore_release import main as restore_release


def service_commands(gpu_enabled: bool):
    commands = []
    if gpu_enabled:
        commands.append(([
            "/opt/retriever/bin/python", "-m", "uvicorn", "production:app",
            "--host", "127.0.0.1", "--port", "7861", "--no-access-log",
        ], "/home/user/app/retriever"))
    commands.append(([
        sys.executable, "-m", "uvicorn", "space_app:app", "--host", "0.0.0.0",
        "--port", "7860", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*",
    ], str(Path(__file__).resolve().parent)))
    return commands


def main():
    gpu_enabled = os.environ.get("NCS_ENABLE_GPU") == "true"
    if gpu_enabled:
        os.environ["BGE_M3_RETRIEVER_URL"] = "http://127.0.0.1:7861"
        os.environ["RERANKER_URL"] = "http://127.0.0.1:7861"
    restore_release()
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    processes = []
    try:
        for command, directory in service_commands(gpu_enabled):
            processes.append(subprocess.Popen(command, cwd=directory))
        while not stopped.wait(1):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A required Space service exited")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
