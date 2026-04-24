from pydantic import BaseModel

import argparse
import sys
import time
from typing import Any, Sequence
import requests


DEFAULT_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_MD = "report.md"
DEFAULT_PROMPTS = [
    "Как лучше подготовиться к собеседованию?",
    "Придумай идею для подарка другу, у которого вроде всё уже есть.",
    "Почему у меня постоянно нет мотивации?",
    "Напиши короткий текст для сторис про путешествие, чтобы звучало не банально.",
    "Можешь помочь составить план на неделю, чтобы я всё успевал?",
    "Объясни, в чём разница между ИИ и машинным обучением.",
    "Хочу начать бегать, но быстро устаю — что я делаю не так?",
    "Подскажи хороший фильм на вечер, что-то не слишком тяжёлое, но интересное.",
    "Как научиться лучше формулировать мысли, особенно когда пишу тексты?",
    "Можешь объяснить простыми словами, что такое квантовая физика?",
]


class Exchange(BaseModel):
    prompt: str
    answer: str
    seconds: float


def ollama_generate_once(
    session: requests.Session,
    base_url: str,
    model: str,
    text_prompt: str,
    timeout: float,
) -> str:
    """
    Отправка запроса на /api/generate

    Args:
        session: HTTP-сессия.
        base_url: корневой url для сервиса.
        model: Тег модели в Ollama.
        text_prompt: Текст промпта.
        timeout: Таймаут ответа.

    Returns:
        Ответ на промт от модели.
    """

    endpoint = base_url + "/api/generate"

    body: dict[str, Any] = {
        "model": model,
        "prompt": text_prompt,
        "stream": False,
    }

    response = session.post(endpoint, json=body, timeout=timeout)

    if response.status_code != 200:
        raise requests.HTTPError("Не удалось получить ответ")
    return response.json()["response"].strip()


def assert_model_available(
    session: requests.Session,
    base_url: str,
    model: str,
    timeout: float,
) -> None:
    """
    Проверка работоспособности сервера и наличия модели

    Args:
        session: HTTP-сессия.
        base_url: корневой url для сервиса.
        model: Тег модели в Ollama.
        timeout: Таймаут ответа.
    """
    tags_url = base_url + "/api/tags"
    response = session.get(tags_url, timeout=timeout)

    if response.status_code != 200:
        raise requests.HTTPError("Не удалось связаться с Ollama")

    model_names: list[str] = []
    for item in response.json().get("models", []):
        name = item.get("name")
        if isinstance(name, str):
            model_names.append(name)

    needed_model_name = model.split(":", 1)[0].lower()
    if not any(name.lower().startswith(needed_model_name) for name in model_names):
        raise ValueError(
            f"Модель «{model}» не найдена. Доступно: {model_names}. Выполните: ollama pull {model}",
        )


def run_prompt_list(
    base_url: str,
    model: str,
    prompts: Sequence[str],
    timeout: float,
) -> list[Exchange]:
    """
    Отправка промптов через Ollama.

    Args:
        base_url: корневой url для сервиса.
        model: Имя модели.
        prompts: Запросы в порядке выполнения.
        timeout: Таймаут ответа.
    Returns:
        Список объектов ``Exchange`` в том же порядке.
    """
    responses: list[Exchange] = []
    prompts_len = len(prompts)

    with requests.Session() as session:
        assert_model_available(session, base_url, model, timeout)

        for prompt_index, prompt in enumerate(prompts, start=1):
            time_start = time.perf_counter()
            model_response = ollama_generate_once(
                session, base_url, model, prompt, timeout
            )
            time_end = time.perf_counter()
            response_duration = time_end - time_start
            responses.append(
                Exchange(
                    prompt=prompt, answer=model_response, seconds=response_duration
                )
            )
            response_preview = model_response
            if len(response_preview) > 200:
                response_preview = response_preview[:160] + "..."
            print(
                f"[{prompt_index}/{prompts_len}]"
                f" {response_duration:.1f} second: {response_preview}",
                flush=True,
            )

    return responses


def escape_md_cell(value: str) -> str:
    """
    Подготовка текста для ячейки Markdown-таблицы.

     Args:
         value: Исходная строка.

     Returns:
         Строка с заменой переносов и экранированием ``|``.
    """
    return value.replace("|", r"\|").replace("\n", "<br>")


def write_markdown_table(path: str, rows: Sequence[Exchange]) -> None:
    """
    Созранение отчёта инференса.

    Args:
        path: Путь к .md файлу.
        rows: Результаты инференса.
        model: Имя модели.
    """
    lines = [
        "# Отчёт инференса (ЛР2 NLP)",
        "",
        "",
        "| № | Запрос к LLM | Вывод LLM |",
        "|---|--------------|-----------|",
    ]
    for index, response in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {escape_md_cell(response.prompt)} | {escape_md_cell(response.answer)} |"
        )
    lines.extend(["", "## Длительность генерации (с)", "", "| № | с |", "|---|---|"])
    for index, response in enumerate(rows, start=1):
        lines.append(f"| {index} | {response.seconds:.2f} |")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def load_prompts_file(path: str) -> list[str]:
    """
    Чтение промптов из файла.

    Args:
        path: Путь к текстовому файлу.

    Returns:
        Список промтов.
    """
    result: list[str] = []
    with open(path) as file:
        for line in file:
            result.append(line.strip())
    return result


def parse_cli() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(description="HTTP-запросы к Ollama.")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Корень Ollama API")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Тег модели")
    parser.add_argument(
        "--timeout", type=float, default=200.0, help="Таймаут одного запроса, с"
    )
    parser.add_argument("--md-out", default=DEFAULT_MD, help="Путь Markdown-отчёта")
    parser.add_argument("--prompts-file", default=None, help="Файл со своими промптами")
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = parse_cli()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)

    responses = run_prompt_list(args.base_url, args.model, prompts, args.timeout)
    write_markdown_table(args.md_out, responses)


if __name__ == "__main__":
    main()
