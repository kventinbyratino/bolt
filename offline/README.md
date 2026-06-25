# Offline dependencies for Windows / Python 3.11

This folder contains a complete offline wheelhouse for `bolt` on:
- Windows `win_amd64`
- CPython `3.11.x`

## Install without bat / ps1 / executable scripts

Run these commands manually in PowerShell from the repository root:

```powershell
python -m pip install --no-index --find-links .\offline\wheels\win_py311 -r .\requirements.txt
python -c "import cadquery, ollama; from OCP.gp import gp_Vec; print('OK')"
```

## Optional helper script

If script execution is allowed in your environment, you can also use:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\install_offline.ps1
```

## Notes

- The wheelhouse includes transitive dependencies required by `cadquery`, `cadquery-ocp`, and `ollama`.
- Target interpreter: Python `3.11.x`.
- The wheels were collected for `win_amd64` only.
- If you use a different Python minor version or architecture, this bundle may not install.
