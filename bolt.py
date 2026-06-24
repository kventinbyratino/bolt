"""
CAD-агент: текст → STEP. Только CadQuery. Без импортов. Кроссплатформенный.
Защита:
  1) AST whitelist: только cadquery, math
  2) Runtime: перехват __import__, exec, eval, compile
  3) Subprocess: запуск с пустым окружением
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import ollama
except ImportError:
    sys.exit("pip install ollama")


# ==================== КОНФИГ ====================
OLLAMA_MODEL = "qwen2.5-coder:7b"
MAX_RETRIES = 3
TIMEOUT_SEC = 60
OUTPUT_DIR = Path("./cad_output")
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_IMPORTS = {"cadquery", "math"}


# ==================== ПРОМПТ ====================
SYSTEM_PROMPT = """Ты — CAD-инженер. Создаёшь 3D-модели ТОЛЬКО через CadQuery → STEP.

СТРОГИЕ ПРАВИЛА:
1. Возвращай ТОЛЬКО Python-код в блоке ```python ... ```. Без пояснений.
2. Разрешено импортировать ТОЛЬКО: cadquery, math. НИЧЕГО ДРУГОГО.
3. ЗАПРЕЩЕНО: import os, sys, subprocess, socket, urllib, requests, shutil, ctypes, exec(), eval(), compile().
4. Финальную модель сохрани в переменную `result`.
5. В конце ОБЯЗАТЕЛЬНО: cq.exporters.export(result, "output.stp")
6. Все размеры в миллиметрах. Параметрический подход.
7. Комментарии на русском.
8. Никаких show(), display(), сетевых вызовов, чтения/записи файлов кроме output.stp.
"""


# ==================== УТИЛИТЫ ====================
def extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def validate_step(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "ISO-10303-21" in f.read(500)
    except Exception:
        return False


# ---------- УРОВЕНЬ 1: AST whitelist ----------
def check_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Синтаксическая ошибка: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Запрещён импорт: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Запрещён импорт: from {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile", "__import__"):
                    return False, f"Запрещён вызов: {node.func.id}()"
    return True, ""


# ---------- УРОВЕНЬ 2: Runtime sandbox ----------
SANDBOX_PREAMBLE = """
import sys as _sys
import builtins as _bi

_orig_import = _bi.__import__
def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ("cadquery", "math"):
        raise ImportError(f"Импорт {name} запрещён")
    return _orig_import(name, globals, locals, fromlist, level)
_bi.__import__ = _restricted_import

def _blocked(*a, **kw):
    raise PermissionError("Операция запрещена sandbox")
_bi.exec = _blocked
_bi.eval = _blocked
_bi.compile = _blocked

for _mod in ("os", "sys", "subprocess", "socket", "urllib", "http", "requests",
             "shutil", "ctypes", "multiprocessing", "threading"):
    class _Blocker:
        def __getattr__(self, name):
            raise PermissionError(f"Модуль {_mod} заблокирован")
    _sys.modules[_mod] = _Blocker()
"""


# ---------- Запуск кода ----------
def run_cad_code(code: str) -> tuple[bool, str, Path | None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        script = tmppath / "cad.py"
        step = tmppath / "output.stp"

        full_code = SANDBOX_PREAMBLE + "\n\n" + code
        script.write_text(full_code, encoding="utf-8")

        safe_env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": ""}
        if sys.platform == "win32":
            for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
                if key in os.environ:
                    safe_env[key] = os.environ[key]

        try:
            res = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True,
                timeout=TIMEOUT_SEC, cwd=str(tmppath), env=safe_env,
            )
            out = res.stdout + ("\n" + res.stderr if res.stderr else "")

            if res.returncode != 0:
                return False, f"Код упал:\n{out}", None
            if not step.exists():
                return False, f"STEP не создан.\n{out}", None
            if not validate_step(step):
                return False, f"Невалидный STEP.\n{out}", None

            idx = len(list(OUTPUT_DIR.glob("*.stp"))) + 1
            final = OUTPUT_DIR / f"model_{idx}.stp"
            step.replace(final)
            return True, f"✅ {final} ({final.stat().st_size / 1024:.1f} KB)", final

        except subprocess.TimeoutExpired:
            return False, f"⏱ Таймаут {TIMEOUT_SEC} сек", None
        except Exception as e:
            return False, f"Ошибка: {e}", None


# ==================== ЯДРО АГЕНТА ====================
def create_model(description: str) -> Path | None:
    print(f"\n🎯 {description}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Создай 3D-модель в STEP:\n{description}"},
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"🤖 Попытка #{attempt}...")
        raw = ollama.chat(model=OLLAMA_MODEL, messages=messages)["message"]["content"]
        code = extract_code(raw)

        if not code:
            print("⚠️ Пустой ответ")
            return None

        ok, reason = check_ast(code)
        if not ok:
            print(f"🚫 AST-блок: {reason}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                "content": f"Нарушение: {reason}. Используй ТОЛЬКО cadquery и math."})
            continue

        print("▶️  Запуск в sandbox...")
        ok, out, path = run_cad_code(code)

        if ok and path:
            print(out)
            return path

        print(f"❌ {out[:400]}\n")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
            "content": f"Ошибка:\n{out}\n\nИсправь код."})

    print("🛑 Лимит попыток исчерпан")
    return None


# ==================== CLI ====================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CAD-агент → STEP")
    p.add_argument("desc", nargs="?", help="Описание изделия")
    args = p.parse_args()

    if args.desc:
        create_model(args.desc)
    else:
        print("🏭 CAD-агент. Введите описание (exit — выход):\n")
        while True:
            try:
                s = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if s.lower() in ("exit", "quit", "выход"):
                break
            if s:
                create_model(s)
                print("\n" + "=" * 60)