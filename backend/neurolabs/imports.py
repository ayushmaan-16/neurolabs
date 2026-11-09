"""Bounded package preview lifecycle. Import only commits after explicit UI selection."""

import base64
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import OrderedDict

from neurolabs.packages import MAX_PACKAGE_BYTES


class ImportBusyError(ValueError):
    pass


class ImportPreviews:
    def __init__(self):
        self.slots = threading.BoundedSemaphore(1)
        self.lock = threading.Lock()
        self.items = OrderedDict()

    def inspect(self, data: bytes, source_format="zip", filename="model.onnx"):
        if source_format not in ("zip", "onnx"):
            raise ValueError("Choose an ONNX model or a neuroLabs ZIP package")
        if len(data) > MAX_PACKAGE_BYTES:
            raise ValueError("Model packages must be at most 6 MiB")
        if not self.slots.acquire(blocking=False):
            raise ImportBusyError("Another model is being checked. Try again when it finishes.")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "neurolabs.package_worker", source_format, filename[:255]],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                stdout, _ = process.communicate(data, timeout=30)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise ValueError("Model compatibility check exceeded its 30 second limit") from exc
            if process.returncode or len(stdout) > MAX_PACKAGE_BYTES * 2:
                raise ValueError(
                    "Model compatibility worker could not complete. Try a smaller package."
                )
            message = json.loads(stdout)
            if "error" in message:
                raise ValueError(message["error"])
            result = message["result"]
            if source_format == "onnx":
                data = base64.b64decode(message["package"], validate=True)
                if len(data) > MAX_PACKAGE_BYTES:
                    raise ValueError("Converted model exceeds the 6 MiB package limit")
            token = secrets.token_hex(16)
            result["token"] = token
            result["expires_in_seconds"] = 900
            with self.lock:
                self._expire()
                while len(self.items) >= 4:
                    self.items.popitem(last=False)
                self.items[token] = (time.monotonic(), data, result)
            return result
        finally:
            self.slots.release()

    def _expire(self):
        expired = [
            key for key, (created, _, _) in self.items.items() if time.monotonic() - created > 900
        ]
        for key in expired:
            del self.items[key]

    def get(self, token: str):
        with self.lock:
            self._expire()
            if token not in self.items:
                raise KeyError("Import preview expired. Select the package again.")
            _, data, result = self.items[token]
            return data, result
