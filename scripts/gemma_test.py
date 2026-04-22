#!/usr/bin/env python3

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_PROMPT = "自己紹介を1文でしてください。"


class ApiError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API request failed with status {status_code}")


def api_request(api_key: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    body = None
    headers = {"x-goog-api-key": api_key}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise ApiError(error.code, error_body) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error: {error.reason}") from error


def list_models(api_key: str) -> list[dict]:
    path = "/models?" + urllib.parse.urlencode({"pageSize": 1000})
    response = api_request(api_key, path)
    return response.get("models", [])


def generate_content(
    api_key: str,
    model: str,
    prompt: str,
    *,
    system_instruction: str | None = None,
    thinking: str = "off",
) -> dict:
    payload: dict = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    if thinking == "high":
        payload["generationConfig"] = {
            "thinkingConfig": {
                "thinkingLevel": "high",
            }
        }

    return api_request(api_key, f"/models/{model}:generateContent", method="POST", payload=payload)


def extract_text(response: dict) -> str:
    texts: list[str] = []

    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)

    return "\n".join(texts).strip()


def print_available_gemma_models(api_key: str) -> None:
    models = list_models(api_key)
    gemma_models = [
        model for model in models if model.get("name", "").startswith("models/gemma-4-")
    ]

    if not gemma_models:
        print("Gemma 4 models were not found in the current API model list.", file=sys.stderr)
        return

    print("Available Gemma 4 models:")
    for model in sorted(gemma_models, key=lambda item: item.get("name", "")):
        name = model.get("name", "")
        display_name = model.get("displayName", "")
        print(f"- {name.replace('models/', '')} ({display_name})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use GEMINI_API_KEY to test access to Gemma 4 on Google AI Studio."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to call (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send to the model",
    )
    parser.add_argument(
        "--system",
        help="Optional system instruction",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "high"],
        default="off",
        help="Enable Gemma 4 thinking mode",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Gemma 4 models and exit",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        return 2

    if args.list_models:
        try:
            print_available_gemma_models(api_key)
        except Exception as error:
            print(f"Failed to list models: {error}", file=sys.stderr)
            return 1
        return 0

    try:
        response = generate_content(
            api_key,
            args.model,
            args.prompt,
            system_instruction=args.system,
            thinking=args.thinking,
        )
    except ApiError as error:
        print(f"API error ({error.status_code}).", file=sys.stderr)
        print(error.body, file=sys.stderr)

        if error.status_code == 404:
            try:
                print(file=sys.stderr)
                print_available_gemma_models(api_key)
            except Exception:
                pass
        return 1
    except Exception as error:
        print(f"Request failed: {error}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    text = extract_text(response)
    if text:
        print(text)
        return 0

    print("The API returned no text content. Full response:", file=sys.stderr)
    print(json.dumps(response, ensure_ascii=False, indent=2), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
