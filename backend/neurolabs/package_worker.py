"""Short-lived native-package validation worker with a parent-enforced deadline."""

import base64
import json
import sys

import torch

from neurolabs.packages import MAX_PACKAGE_BYTES, inspect_package


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        data = sys.stdin.buffer.read(MAX_PACKAGE_BYTES + 1)
        if len(sys.argv) > 1 and sys.argv[1] == "onnx":
            from neurolabs.onnx_import import convert_onnx

            package, report = convert_onnx(data, sys.argv[2] if len(sys.argv) > 2 else "model.onnx")
            result = {"result": report, "package": base64.b64encode(package).decode("ascii")}
        else:
            result = {"result": inspect_package(data)}
            result["result"]["source_format"] = "zip"
            result["result"]["verification"]["method"] = "supplied_examples"
    except Exception as exc:
        result = {"error": str(exc)[:2000]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
