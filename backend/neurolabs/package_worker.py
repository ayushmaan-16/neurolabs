"""Short-lived native-package validation worker with a parent-enforced deadline."""

import json
import sys

import torch

from neurolabs.packages import MAX_PACKAGE_BYTES, inspect_package


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        data = sys.stdin.buffer.read(MAX_PACKAGE_BYTES + 1)
        result = {"result": inspect_package(data)}
    except Exception as exc:
        result = {"error": str(exc)[:2000]}
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
