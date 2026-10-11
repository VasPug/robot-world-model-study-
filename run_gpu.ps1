$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
Push-Location $PSScriptRoot
try {
    uv run --project gpu_extension --locked python -m gpu_extension.study @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
