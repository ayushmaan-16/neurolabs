import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { randomUUID } from "node:crypto";

const python =
  process.platform === "win32"
    ? ".venv/Scripts/python.exe"
    : ".venv/bin/python";
export default defineConfig({
  testDir: "./frontend/tests",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:8001",
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `"${python}" -m uvicorn neurolabs.api:create_app --factory --host 127.0.0.1 --port 8001`,
    url: "http://127.0.0.1:8001/api/v1/health",
    reuseExistingServer: false,
    timeout: 60_000,
    env: { NEUROLABS_DATA_DIR: resolve(".local", `e2e-${randomUUID()}`) },
  },
});
