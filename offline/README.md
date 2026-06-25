# Offline dependencies for Windows / Python 3.11

This folder contains a complete offline wheelhouse for `bolt` on:
- Windows `win_amd64`
- CPython `3.11.x`

## Install

Run in PowerShell from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\install_offline.ps1
```

Or manually:

```powershell
python -m pip install --no-index --find-links .\offline\wheels\win_py311 -r .\requirements.txt
python -c "import cadquery, ollama; from OCP.gp import gp_Vec; print('OK')"
```

## Notes

- The wheelhouse includes transitive dependencies required by `cadquery`, `cadquery-ocp`, and `ollama`.
- Target interpreter: Python `3.11.x`.
- The wheels were collected for `win_amd64` only.
- If you use a different Python minor version or architecture, this bundle may not install.
