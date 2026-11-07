"""Repeatable warm CPU benchmark. Excludes HTTP, storage, startup and browser rendering."""

import json
import platform
import statistics
import time

import torch
from neurolabs.runtime import execute
from neurolabs.samples import samples
from neurolabs.schema import Activation, Dense, ModelSpec, TensorSpec


def benchmark(spec, parameters, inputs, iterations=50):
    for _ in range(5):
        execute(spec, parameters, inputs, "benchmark")
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        execute(spec, parameters, inputs, "benchmark")
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "model": spec.id,
        "parameters": sum(value.numel() for value in parameters.values()),
        "batch": len(inputs),
        "iterations": iterations,
        "p50_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(0.95 * (iterations - 1))], 3),
    }


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    results = [benchmark(spec, params, [inputs]) for spec, params, inputs in samples()]
    tensors, operations, parameters = [], [], {}
    current = "input"
    for index in range(15):
        name = f"dense{index}"
        tensors.append(TensorSpec(id=f"{name}.weight", shape=(256, 256)))
        parameters[f"{name}.weight"] = torch.randn(256, 256) / 16
        operations.extend(
            [
                Dense(id=name, input=current, output=f"{name}.z", weight=f"{name}.weight"),
                Activation(kind="relu", id=f"relu{index}", input=f"{name}.z", output=f"{name}.a"),
            ]
        )
        current = f"{name}.a"
    spec = ModelSpec(
        id="near-limit",
        name="Near-limit dense chain",
        input_width=256,
        output_name=current,
        tensors=tuple(tensors),
        operations=tuple(operations),
    )
    results.append(benchmark(spec, parameters, torch.randn(1, 256).tolist()))
    results.append(benchmark(spec, parameters, torch.randn(256, 256).tolist(), iterations=10))
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "threads": 1,
                "seed": 42,
                "capture": "none",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
