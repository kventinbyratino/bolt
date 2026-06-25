$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $root '..')
$wheelDir = Join-Path $repoRoot 'offline/wheels/win_py311'
$requirements = Join-Path $repoRoot 'requirements.txt'

if (-not (Test-Path $wheelDir)) {
    throw "Wheel directory not found: $wheelDir"
}
if (-not (Test-Path $requirements)) {
    throw "requirements.txt not found: $requirements"
}

python -m pip install --no-index --find-links "$wheelDir" -r "$requirements"
python -c "import cadquery, ollama; from OCP.gp import gp_Vec; print('OK')"
