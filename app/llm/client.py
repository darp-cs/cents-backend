from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


class LLMClientError(RuntimeError):
    pass


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.llm_service_api_key.strip():
        headers["X-API-Key"] = settings.llm_service_api_key.strip()
    return headers


def _build_endpoint(path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{settings.llm_service_base_url.rstrip('/')}{normalized_path}"


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail

    return str(payload)


async def generate_text(payload: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout=settings.llm_service_timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _build_endpoint(settings.llm_service_generate_path),
                json=payload,
                headers=_build_headers(),
            )
    except httpx.TimeoutException as exc:
        raise LLMClientError("LLM service request timed out.") from exc
    except httpx.RequestError as exc:
        raise LLMClientError(f"LLM service connection failed: {exc}") from exc

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        raise LLMClientError(f"LLM service error ({response.status_code}): {detail}")

    body = response.json()
    if not isinstance(body, dict):
        raise LLMClientError("LLM service returned an invalid response payload.")

    return body


async def list_models() -> list[str]:
    payload = await list_models_payload()
    models = payload.get("models", [])
    return [str(model).strip() for model in models if str(model).strip()]


def _normalize_model_folders(raw_folders: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_folders, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_config in raw_folders.items():
        folder_name = str(raw_name).strip()
        if not folder_name or not isinstance(raw_config, dict):
            continue

        raw_models = raw_config.get("models", [])
        models = [str(model).strip() for model in raw_models if str(model).strip()] if isinstance(raw_models, list) else []
        default_model_raw = raw_config.get("default_model")
        default_model = (
            str(default_model_raw).strip() if isinstance(default_model_raw, str) and default_model_raw.strip() else None
        )

        normalized[folder_name] = {
            "models": models,
            "default_model": default_model,
        }

    return normalized


async def list_models_payload() -> dict[str, Any]:
    timeout = httpx.Timeout(timeout=settings.llm_service_timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                _build_endpoint(settings.llm_service_models_path),
                headers=_build_headers(),
            )
    except httpx.TimeoutException as exc:
        raise LLMClientError("LLM models request timed out.") from exc
    except httpx.RequestError as exc:
        raise LLMClientError(f"LLM service connection failed: {exc}") from exc

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        raise LLMClientError(f"LLM models request failed ({response.status_code}): {detail}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise LLMClientError("LLM models response payload is invalid.")

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise LLMClientError("LLM models response is missing a valid models list.")

    return {
        "models": [str(model).strip() for model in models if str(model).strip()],
        "folders": _normalize_model_folders(payload.get("folders", {})),
    }
