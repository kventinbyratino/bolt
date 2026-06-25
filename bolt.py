"""
CAD-агент: текст → STEP. Только CadQuery. Безопасный запуск через sandbox.

Новый поток:
  1) Пользователь вводит запрос.
  2) Ollama сначала уточняет/нормализует запрос и формирует точный prompt.
  3) Ollama генерирует CadQuery-код.
  4) Код проходит AST-проверку и исполняется в sandbox.

Все артефакты сохраняются рядом с bolt.py:
  - cad_output/      -> финальные STEP
  - sandbox_runs/    -> временные и sandbox-файлы по каждому запуску
  - bolt.log         -> общий подробный лог
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


# ==================== ПУТИ ====================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "cad_output"
RUNS_DIR = BASE_DIR / "sandbox_runs"
APP_LOG_PATH = BASE_DIR / "bolt.log"

OUTPUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


# ==================== КОНФИГ ====================
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_REFINER_MODEL = DEFAULT_OLLAMA_MODEL
MAX_RETRIES = 3
TIMEOUT_SEC = 60
ALLOWED_IMPORTS = {"cadquery", "math"}


# ==================== PROMPTS ====================
REFINER_SYSTEM_PROMPT = """Ты — CAD-аналитик. Твоя задача: превратить краткий запрос пользователя в точное техническое задание для генерации 3D-модели через CadQuery.

Правила:
1. Не пиши код.
2. Не задавай пользователю вопросов.
3. Если данных не хватает, аккуратно прими разумные инженерные допущения и явно перечисли их.
4. Ответ должен быть на русском языке.
5. Сформируй результат строго в JSON-объекте такого вида:
{
  "refined_request": "краткое уточнённое описание изделия",
  "assumptions": ["...", "..."],
  "cad_prompt": "готовый точный prompt для генератора CadQuery-кода"
}
6. В поле cad_prompt опиши:
   - геометрию,
   - размеры в мм,
   - порядок построения,
   - ключевые параметры,
   - что должно быть сохранено в переменную result,
   - что нужен экспорт в output.stp.
7. Никаких markdown-блоков, только JSON.
"""

CAD_SYSTEM_PROMPT = """Ты — CAD-инженер. Создаёшь 3D-модели ТОЛЬКО через CadQuery → STEP.

СТРОГИЕ ПРАВИЛА:
1. Возвращай ТОЛЬКО Python-код в блоке ```python ...```. Без пояснений.
2. Разрешено импортировать ТОЛЬКО: cadquery, math. НИЧЕГО ДРУГОГО.
3. ЗАПРЕЩЕНО: import os, sys, subprocess, socket, urllib, requests, shutil, ctypes, pathlib, json, tempfile, threading, multiprocessing, builtins.
4. ЗАПРЕЩЕНО: exec(), eval(), compile(), __import__().
5. Финальную модель сохрани в переменную `result`.
6. В конце ОБЯЗАТЕЛЬНО: cq.exporters.export(result, "output.stp")
7. Все размеры в миллиметрах. Параметрический подход.
8. Комментарии на русском.
9. Никаких show(), display(), сетевых вызовов, чтения/записи файлов кроме output.stp.
10. Если нужны константы — объяви их явно в коде.
"""


# ==================== ЛОГИРОВАНИЕ ====================
def configure_root_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("bolt")
    if logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))

    file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.debug("Root logger configured. log_file=%s", APP_LOG_PATH)
    return logger


def attach_run_logger(run_dir: Path) -> logging.Handler:
    logger = logging.getLogger("bolt")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.debug("Attached run logger: %s", run_dir / "run.log")
    return handler


logger = configure_root_logger()


# ==================== УТИЛИТЫ ====================
def get_ollama_client():
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Не найден пакет ollama. Установите в окружение проекта: .venv/bin/python -m pip install ollama"
        ) from exc
    return ollama


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    inline = re.search(r"(\{.*\})", text, re.DOTALL)
    if inline:
        return json.loads(inline.group(1))

    raise ValueError("Не удалось извлечь JSON из ответа Ollama")


def create_run_dir(description: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "_", description.strip())[:40].strip("_") or "request"
    run_id = f"{timestamp}_{slug}_{uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_step(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "STEP-файл не создан"

    size = path.stat().st_size
    if size < 1000:
        return False, f"STEP-файл слишком маленький: {size} байт"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            head = file.read(500)
    except Exception as exc:
        return False, f"Не удалось прочитать STEP: {exc}"

    if "ISO-10303-21" not in head:
        return False, "В STEP не найден сигнатурный заголовок ISO-10303-21"

    return True, f"STEP валиден, размер {size} байт"


# ---------- УРОВЕНЬ 1: AST whitelist ----------
def check_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Синтаксическая ошибка: {exc}"

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
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "compile", "__import__"}:
                return False, f"Запрещён вызов: {node.func.id}()"

    return True, "AST-проверка пройдена"


# ---------- УРОВЕНЬ 2: Runtime sandbox ----------
SANDBOX_PREAMBLE = """
import sys as _sys
import builtins as _bi

_ALLOWED_ROOTS = {"cadquery", "math"}
_ALLOWED_INTERNALS = set(_sys.builtin_module_names) | {
    "_io", "io", "_frozen_importlib", "_frozen_importlib_external",
    "zipimport", "encodings", "codecs", "importlib", "marshal",
    "posix", "nt", "time", "_thread", "atexit"
}

_orig_import = _bi.__import__
def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    caller_name = ""
    if globals:
        caller_name = globals.get("__package__") or globals.get("__name__") or ""
    caller_root = caller_name.split(".")[0] if caller_name else ""

    if root in _ALLOWED_ROOTS:
        return _orig_import(name, globals, locals, fromlist, level)
    if caller_root in _ALLOWED_ROOTS:
        return _orig_import(name, globals, locals, fromlist, level)
    if caller_name and caller_name != "__main__":
        return _orig_import(name, globals, locals, fromlist, level)
    if root in _ALLOWED_INTERNALS or root.startswith("_"):
        return _orig_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Импорт {name} запрещён")
_bi.__import__ = _restricted_import

def _blocked(*a, **kw):
    raise PermissionError("Операция запрещена sandbox")

_orig_eval = _bi.eval

def _restricted_eval(*args, **kwargs):
    frame = _sys._getframe(1)
    caller_globals = frame.f_globals if frame else {}
    caller_name = caller_globals.get("__package__") or caller_globals.get("__name__") or ""
    if caller_name and caller_name != "__main__":
        return _orig_eval(*args, **kwargs)
    raise PermissionError("Операция запрещена sandbox")

_bi.eval = _restricted_eval
"""


def build_child_env() -> dict[str, str]:
    safe_env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "",
    }
    if sys.platform == "win32":
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
            if key in os.environ:
                safe_env[key] = os.environ[key]
    return safe_env


def run_cad_code(code: str, run_dir: Path, timeout_sec: int) -> tuple[bool, str, Path | None]:
    script_path = run_dir / "cad_generated.py"
    step_path = run_dir / "output.stp"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    full_code = SANDBOX_PREAMBLE + "\n\n" + code

    safe_write_text(script_path, full_code)
    logger.info("Запуск sandbox: %s", script_path)
    logger.debug("Sandbox env keys: %s", sorted(build_child_env().keys()))

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(run_dir),
            env=build_child_env(),
        )
    except subprocess.TimeoutExpired:
        logger.error("Sandbox timeout: %s sec", timeout_sec)
        safe_write_text(stdout_path, "")
        safe_write_text(stderr_path, f"Timeout after {timeout_sec} seconds")
        return False, f"⏱ Таймаут {timeout_sec} сек", None
    except Exception as exc:
        logger.exception("Ошибка запуска sandbox")
        safe_write_text(stdout_path, "")
        safe_write_text(stderr_path, traceback.format_exc())
        return False, f"Ошибка запуска sandbox: {exc}", None

    safe_write_text(stdout_path, result.stdout)
    safe_write_text(stderr_path, result.stderr)

    logger.debug("Sandbox returncode=%s", result.returncode)
    logger.debug("Sandbox stdout:\n%s", result.stdout.strip())
    logger.debug("Sandbox stderr:\n%s", result.stderr.strip())

    if result.returncode != 0:
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return False, f"Код упал:\n{output.strip()}", None

    valid, reason = validate_step(step_path)
    logger.info("STEP validation: %s", reason)
    if not valid:
        return False, f"Невалидный STEP: {reason}", None

    index = len(list(OUTPUT_DIR.glob("*.stp"))) + 1
    final_path = OUTPUT_DIR / f"model_{index}.stp"
    step_path.replace(final_path)
    logger.info("STEP перенесён в %s", final_path)
    return True, f"✅ {final_path} ({final_path.stat().st_size / 1024:.1f} KB)", final_path


# ==================== OLLAMA FLOW ====================
def ollama_chat(model: str, messages: list[dict[str, str]]) -> str:
    client = get_ollama_client()
    logger.info("Ollama request → model=%s, messages=%s", model, len(messages))
    response = client.chat(model=model, messages=messages)
    content = response["message"]["content"]
    logger.debug("Ollama raw response:\n%s", content)
    return content


def refine_request(description: str, run_dir: Path, model: str) -> dict[str, Any]:
    logger.info("Шаг 1/2: уточнение запроса через Ollama")
    messages = [
        {"role": "system", "content": REFINER_SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    raw = ollama_chat(model=model, messages=messages)
    safe_write_text(run_dir / "01_refiner_raw.txt", raw)

    refined = extract_json_object(raw)
    safe_write_json(run_dir / "02_refined_request.json", refined)

    for key in ("refined_request", "assumptions", "cad_prompt"):
        if key not in refined:
            raise ValueError(f"В ответе уточнителя нет обязательного поля: {key}")

    logger.info("Уточнённый запрос: %s", refined["refined_request"])
    assumptions = refined.get("assumptions") or []
    if assumptions:
        logger.info("Допущения: %s", "; ".join(str(x) for x in assumptions))
    return refined


def generate_cad_code(refined: dict[str, Any], run_dir: Path, model: str, attempt: int) -> tuple[str, str]:
    logger.info("Шаг 2/2: генерация CadQuery-кода, попытка #%s", attempt)
    cad_prompt = str(refined["cad_prompt"]).strip()
    safe_write_text(run_dir / f"10_attempt_{attempt}_cad_prompt.txt", cad_prompt)

    messages = [
        {"role": "system", "content": CAD_SYSTEM_PROMPT},
        {"role": "user", "content": cad_prompt},
    ]
    raw = ollama_chat(model=model, messages=messages)
    safe_write_text(run_dir / f"11_attempt_{attempt}_raw_response.txt", raw)

    code = extract_code(raw)
    safe_write_text(run_dir / f"12_attempt_{attempt}_extracted_code.py", code)
    return raw, code


def build_final_report(run_dir: Path, status: str, final_message: str) -> Path:
    report_path = run_dir / "final_report.md"
    sections = [
        "# Итоговый отчёт запуска",
        "",
        f"- run_dir: `{run_dir}`",
        f"- status: `{status}`",
        f"- final_message: {final_message}",
        "",
        "## Артефакты",
        "",
    ]

    artifact_paths = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and path.name != report_path.name
    )

    for path in artifact_paths:
        rel = path.relative_to(run_dir)
        sections.append(f"- `{rel}` ({path.stat().st_size} bytes)")

    sections.append("")
    sections.append("## Логи и ошибки")
    sections.append("")

    for path in artifact_paths:
        rel = path.relative_to(run_dir)
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".log", ".json", ".py", ".md"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            content = f"<не удалось прочитать: {exc}>"
        sections.append(f"### `{rel}`")
        sections.append("")
        sections.append("```")
        sections.append(content.rstrip())
        sections.append("```")
        sections.append("")

    safe_write_text(report_path, "\n".join(sections).rstrip() + "\n")
    return report_path


def print_refined_summary(source_request: str, refined: dict[str, Any]) -> None:
    print(f"\n🎯 Исходный запрос: {source_request}")
    print(f"🧭 Уточнённый запрос: {refined['refined_request']}")
    assumptions = refined.get("assumptions") or []
    if assumptions:
        print("📌 Допущения:")
        for item in assumptions:
            print(f"   - {item}")


def confirm_or_revise_refinement(
    original_description: str,
    refined: dict[str, Any],
    run_dir: Path,
    model: str,
    interactive: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    current_source = original_description
    current_refined = refined
    revision_index = 0

    while True:
        print_refined_summary(current_source, current_refined)
        if not interactive:
            logger.info("Подтверждение уточнённого запроса пропущено: неинтерактивный режим")
            return current_source, current_refined

        print("\nПодтвердите уточнение:")
        print("  Enter / yes / y  — принять")
        print("  edit             — скорректировать запрос")
        print("  cancel           — отменить запуск")
        answer = input("confirm> ").strip()
        normalized = answer.lower()

        if normalized in {"", "y", "yes", "да", "ок", "accept"}:
            logger.info("Пользователь подтвердил уточнённый запрос")
            safe_write_text(run_dir / "03_confirmation_status.txt", "accepted")
            return current_source, current_refined

        if normalized in {"cancel", "c", "n", "no", "нет", "отмена"}:
            logger.info("Пользователь отменил запуск после уточнения")
            safe_write_text(run_dir / "03_confirmation_status.txt", "cancelled")
            return None, None

        if normalized == "edit":
            edited = input("Введите скорректированный запрос> ").strip()
            if not edited:
                print("⚠️ Пустая корректировка, оставляю текущий вариант.")
                continue

            revision_index += 1
            current_source = edited
            safe_write_text(run_dir / f"03_revision_{revision_index:02d}_user_request.txt", edited)
            logger.info("Пользователь внёс корректировку #%s: %s", revision_index, edited)
            current_refined = refine_request(edited, run_dir, model=model)
            safe_write_json(run_dir / f"03_revision_{revision_index:02d}_refined_request.json", current_refined)
            continue

        print("⚠️ Не понял ответ. Используйте Enter/yes, edit или cancel.")


# ==================== ЯДРО АГЕНТА ====================
def create_model(
    description: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    refiner_model: str = DEFAULT_REFINER_MODEL,
    timeout_sec: int = TIMEOUT_SEC,
    max_retries: int = MAX_RETRIES,
) -> Path | None:
    run_dir = create_run_dir(description)
    run_handler = attach_run_logger(run_dir)
    result_path: Path | None = None
    final_status = "unknown"
    final_message = "Запуск завершён без итогового статуса"

    try:
        logger.info("=" * 80)
        logger.info("Новый запуск: %s", run_dir.name)
        logger.info("Базовая папка: %s", BASE_DIR)
        logger.info("Папка запуска: %s", run_dir)
        logger.info("Исходный запрос: %s", description)
        logger.info("Модель генерации: %s | модель уточнения: %s", model, refiner_model)
        logger.info("Таймаут sandbox: %s сек | max_retries=%s", timeout_sec, max_retries)

        safe_write_text(run_dir / "00_user_request.txt", description)

        interactive = sys.stdin.isatty()
        refined = refine_request(description, run_dir, model=refiner_model)
        confirmed_description, confirmed_refined = confirm_or_revise_refinement(
            original_description=description,
            refined=refined,
            run_dir=run_dir,
            model=refiner_model,
            interactive=interactive,
        )
        if not confirmed_description or not confirmed_refined:
            final_status = "cancelled"
            final_message = "Запуск отменён пользователем после этапа уточнения"
            print("🛑 Запуск отменён пользователем")
            return None

        description = confirmed_description
        refined = confirmed_refined

        for attempt in range(1, max_retries + 1):
            logger.info("Попытка генерации #%s", attempt)
            print(f"🤖 Попытка #{attempt}...")
            raw, code = generate_cad_code(refined, run_dir, model=model, attempt=attempt)

            if not code:
                logger.error("Пустой код от Ollama")
                final_status = "failed"
                final_message = "Пустой ответ от Ollama при генерации CadQuery-кода"
                print("⚠️ Пустой ответ")
                return None

            ok, reason = check_ast(code)
            safe_write_text(run_dir / f"13_attempt_{attempt}_ast_check.txt", reason)
            if not ok:
                logger.warning("AST-блок на попытке %s: %s", attempt, reason)
                print(f"🚫 AST-блок: {reason}")

                repair_prompt = textwrap.dedent(
                    f"""
                    Нарушение sandbox/AST: {reason}

                    Перегенерируй код строго по правилам:
                    - только cadquery и math
                    - финальная модель в result
                    - экспорт через cq.exporters.export(result, \"output.stp\")
                    - без любых посторонних импортов и опасных вызовов
                    """
                ).strip()

                refined["cad_prompt"] = f"{refined['cad_prompt']}\n\nИсправление после AST-ошибки:\n{repair_prompt}"
                safe_write_json(run_dir / f"14_attempt_{attempt}_refined_after_ast.json", refined)
                final_status = "retrying_after_ast_error"
                final_message = reason
                continue

            print("▶️ Запуск в sandbox...")
            ok, out, path = run_cad_code(code, run_dir=run_dir, timeout_sec=timeout_sec)
            safe_write_text(run_dir / f"15_attempt_{attempt}_sandbox_result.txt", out)

            if ok and path:
                logger.info("Успешное завершение: %s", path)
                final_status = "success"
                final_message = out
                result_path = path
                print(out)
                return path

            logger.warning("Ошибка sandbox на попытке %s: %s", attempt, out)
            print(f"❌ {out[:600]}\n")

            refined["cad_prompt"] = (
                f"{refined['cad_prompt']}\n\n"
                f"Предыдущая попытка завершилась ошибкой sandbox:\n{out}\n"
                "Исправь CadQuery-код, сохрани ту же цель модели и соблюди все ограничения."
            )
            safe_write_json(run_dir / f"16_attempt_{attempt}_refined_after_error.json", refined)
            final_status = "retrying_after_sandbox_error"
            final_message = out

        logger.error("Лимит попыток исчерпан")
        final_status = "failed"
        final_message = "Лимит попыток генерации исчерпан"
        print("🛑 Лимит попыток исчерпан")
        return None

    except Exception as exc:
        logger.exception("Критическая ошибка в create_model")
        safe_write_text(run_dir / "99_fatal_error.txt", traceback.format_exc())
        final_status = "fatal_error"
        final_message = f"Критическая ошибка: {exc}"
        print(f"💥 Ошибка: {exc}")
        return None
    finally:
        logging.getLogger("bolt").removeHandler(run_handler)
        run_handler.close()
        report_path = build_final_report(run_dir, status=final_status, final_message=final_message)
        print(f"📝 Итоговый отчёт: {report_path}")
        if result_path:
            logger.info("Финальный отчёт сохранён: %s", report_path)


# ==================== CLI ====================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD-агент → STEP через Ollama + CadQuery")
    parser.add_argument("desc", nargs="?", help="Описание изделия")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Модель Ollama для генерации CadQuery-кода")
    parser.add_argument("--refiner-model", default=DEFAULT_REFINER_MODEL, help="Модель Ollama для уточнения запроса")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SEC, help="Таймаут sandbox в секундах")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="Максимум попыток генерации")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень логирования")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_root_logger(args.log_level)
    logger.info("CLI args: %s", vars(args))

    if args.desc:
        path = create_model(
            args.desc,
            model=args.model,
            refiner_model=args.refiner_model,
            timeout_sec=args.timeout,
            max_retries=args.retries,
        )
        return 0 if path else 1

    print("🏭 CAD-агент. Введите описание (exit — выход):\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if user_input.lower() in {"exit", "quit", "выход"}:
            return 0

        if user_input:
            create_model(
                user_input,
                model=args.model,
                refiner_model=args.refiner_model,
                timeout_sec=args.timeout,
                max_retries=args.retries,
            )
            print("\n" + "=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
