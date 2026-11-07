$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Python environment is missing. Run uv sync --locked first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'frontend\dist\index.html'))) {
    throw 'Browser build is missing. Run npm ci and npm run build first.'
}
Write-Host 'neuroLabs: http://127.0.0.1:8000 (Ctrl+C to stop)'
& $projectPython -m uvicorn neurolabs.api:create_app --factory --host 127.0.0.1 --port 8000
