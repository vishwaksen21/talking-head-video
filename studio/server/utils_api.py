"""Client for the DCAS Utilities gateway.

Contract, read from the gateway's live `/openapi.json` and confirmed by calling it:

    GET  /llm/models?location=<loc>      -> {location, count, models: [{id, name, ...}]}
    POST /llm/generate                   -> {text, candidates, usage, model, finish_reason}
    POST /llm/videos  {model, prompt, config, location}
                                         -> {operation, done, status, videos, error, rai_*}
    GET  /llm/videos?operation=<name>    -> same shape, poll until done
    Auth: `X-API-Key` header. Missing or wrong key -> 401 {"detail": "unauthorized"}.

`config` is passed to Google's GenerateVideosConfig unmodified, so parameters use Google's
names. Finished videos come back inline as base64 (`videos[].data`).

Retries: idempotent calls (GET, and transcription, which only costs tokens) retry on 429,
502/503/504 and network errors with backoff, honouring Retry-After. Starting a video is
retried ONLY on 429 or a failed connection, where the gateway provably did not accept the
job - retrying after a 5xx could start (and bill) a second generation.

The API key never leaves this module: it is not logged and not included in errors.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

log = logging.getLogger("studio.utils_api")

RETRYABLE_STATUS = {429, 502, 503, 504}


class UtilsAPIError(Exception):
    """A failure talking to the gateway, phrased for the person using the app."""

    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ProviderCallsDisabled(UtilsAPIError):
    """Raised instead of making any network call while the provider gate is off."""

    def __init__(self):
        super().__init__("Provider calls are disabled (STUDIO_PROVIDER_CALLS is off). Previews still work; set "
                         "STUDIO_PROVIDER_CALLS=on in .env and restart to allow generation.")


def gated_client(settings, **kwargs) -> "UtilsClient":
    """The only way the app builds a real client: refuses while the gate is off."""
    if not getattr(settings, "provider_calls", False):
        raise ProviderCallsDisabled()
    return UtilsClient(settings.base_url, settings.api_key, timeout=settings.request_timeout, **kwargs)


@dataclass
class VideoOperation:
    operation: str
    done: bool
    status: str
    videos: list
    error: Optional[str]
    rai_filtered_count: Optional[int]
    rai_filtered_reasons: Optional[list]

    @classmethod
    def from_body(cls, body: dict) -> "VideoOperation":
        return cls(
            operation=str(body.get("operation") or ""),
            done=bool(body.get("done")),
            status=str(body.get("status") or "").lower(),
            videos=list(body.get("videos") or []),
            error=body.get("error"),
            rai_filtered_count=body.get("rai_filtered_count"),
            rai_filtered_reasons=body.get("rai_filtered_reasons"),
        )


def decode_base64(data: str) -> bytes:
    """Decode standard or URL-safe base64 (this gateway has been seen to use both).

    `base64.b64decode` silently drops characters outside its alphabet, which turns
    URL-safe input into corrupt bytes, so the alphabet is normalised first.
    """
    data = data.strip().replace("-", "+").replace("_", "/")
    data += "=" * (-len(data) % 4)
    return base64.b64decode(data, validate=True)


def _detail(response: httpx.Response) -> str:
    if "html" in response.headers.get("content-type", ""):
        return "the server answered with an HTML error page, not the API"
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if isinstance(detail, list):  # FastAPI validation errors
            return "; ".join(str(d.get("msg", d)) for d in detail)[:300]
        if detail:
            return str(detail)[:300]
    return str(body)[:300]


class UtilsClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
        sleep=time.sleep,
    ):
        if not base_url:
            raise UtilsAPIError("No Utilities API base URL is configured.")
        if not api_key:
            raise UtilsAPIError("No Utilities API key is configured.")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=httpx.Timeout(timeout, connect=15.0),
            transport=transport,
        )
        self._max_retries = max_retries
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, *, retry_5xx: bool, **kwargs) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.ConnectError as exc:
                # The request never reached the gateway, so retrying is always safe.
                if attempt <= self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise UtilsAPIError(
                    "Could not reach the Utilities API. Check the base URL and your network.",
                    retryable=True,
                ) from exc
            except httpx.TimeoutException as exc:
                if retry_5xx and attempt <= self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise UtilsAPIError("The Utilities API did not answer in time.", retryable=True) from exc
            except httpx.HTTPError as exc:
                raise UtilsAPIError(f"Network error talking to the Utilities API ({exc.__class__.__name__}).") from exc

            status = response.status_code
            if status < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise UtilsAPIError("The Utilities API returned a response that is not JSON.", status=status) from exc

            retryable = status == 429 or (retry_5xx and status in RETRYABLE_STATUS)
            if retryable and attempt <= self._max_retries:
                log.warning("utils %s %s -> %s, retrying (attempt %d)", method, path, status, attempt)
                self._backoff(attempt, response.headers.get("retry-after"))
                continue
            raise self._error_for(status, _detail(response))

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        delay = min(2 ** attempt, 30)
        if retry_after:
            try:
                delay = min(float(retry_after), 60)
            except ValueError:
                pass
        self._sleep(delay)

    @staticmethod
    def _error_for(status: int, detail: str) -> UtilsAPIError:
        if status == 401:
            return UtilsAPIError("The API key was rejected (401 unauthorized). Check the key.", status=status)
        if status == 403:
            return UtilsAPIError(f"This key is not allowed to use that resource (403): {detail}", status=status)
        if status == 404:
            return UtilsAPIError(f"Not found (404): {detail}. Check the base URL.", status=status)
        if status == 429:
            return UtilsAPIError("Rate limit reached (429). Wait a minute and try again.", status=status, retryable=True)
        if status in (400, 422):
            return UtilsAPIError(f"The API refused the request: {detail}", status=status)
        return UtilsAPIError(f"Utilities API error {status}: {detail}", status=status, retryable=status >= 500)

    # ------------------------------------------------------------------ endpoints
    def list_models(self, location: str) -> list[dict]:
        body = self._request("GET", "/llm/models", params={"location": location}, retry_5xx=True)
        return list(body.get("models") or [])

    def start_video(self, *, model: str, prompt: str, config: dict, location: str) -> VideoOperation:
        body = self._request(
            "POST",
            "/llm/videos",
            json={"model": model, "prompt": prompt, "config": config, "location": location},
            retry_5xx=False,
        )
        op = VideoOperation.from_body(body)
        if not op.operation and not op.done:
            raise UtilsAPIError("The API accepted the request but returned no operation to track.")
        return op

    def poll_video(self, operation: str) -> VideoOperation:
        body = self._request("GET", "/llm/videos", params={"operation": operation}, retry_5xx=True)
        return VideoOperation.from_body(body)

    def generate(self, body: dict) -> dict:
        return self._request("POST", "/llm/generate", json=body, retry_5xx=True)
