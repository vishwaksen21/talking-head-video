# Reconnaissance: video direction and providers (before changes)

Checked on 3 October 2026 using local files only. No network calls were made.

## Current behaviour

| Area | Where | What it does |
|---|---|---|
| Strict prompt | `studio/server/strict.py` → `build_prompt(script, speech_seconds, reference_mode)` | Builds one hard-coded English paragraph. It covers identity preservation, background (keep the reference's or plain grey), "gentle head movement", static camera, no cuts or other people, and the exact script quoted. |
| Negative prompt | `strict.py` `NEGATIVE_PROMPT` / `NEGATIVE_PROMPT_FULL`; chosen in `main.py` `create_strict` | A fixed string per reference mode, stored in `params.video_config`. |
| Creative prompt | Creative mode, `main.py` `create_project` | The user's text is sent verbatim. |
| Veo request | `utils_api.UtilsClient.start_video` | Sends `POST /llm/videos {model, prompt, config, location}`. `config` comes from `catalog.build_video_config`, plus `reference_images` from `pipeline._scene_config`. |
| Reference image to Veo | `pipeline._scene_config` | `config.reference_images = [{image: {image_bytes(b64), mime_type}, reference_type: "asset"}]`. The full upload by default; a face crop as an option. |
| Capabilities | `catalog.py` | A table per Veo family: durations, aspect ratios, resolutions, audio, references. |
| Realism instructions | Inside `build_prompt` only | "natural eye contact and natural blinking, subtle natural expressions and gentle head movement". No structured control. |
| Network calls | `UtilsClient._request` | Used by `list_models`, `start_video`, `poll_video` and `generate`. No global switch to disable them. |
| Tests | `studio/tests/` | 96 offline tests using mocked transports. One opt-in live test is skipped. |

## What "Omni" is

- **Not used anywhere in this repository.** No code, configuration or UI refers to it.
- **Model ids listed by the gateway** (earlier local copies of `GET /llm/models`):
  - `gemini-omni-1.1-flash-preview`: at us-central1 and global
  - `gemini-omni-flash-preview`: at us-central1 only
- **Endpoint the gateway offers for Gemini models:** `POST /llm/generate {model, contents, system_instruction?, config, location}`. It is `additionalProperties: false`, and `config` is forwarded to `GenerateContentConfig` unmodified. Verified from the local copy of the gateway's `/openapi.json`.
- **Response:** `{text, candidates, usage, model, finish_reason}`. **There is no video field.** `POST /llm/videos` documents its `model` as a "Veo model".
- **Authentication:** the same `X-API-Key` header as Veo.
- **Not verifiable locally:**
  - whether Omni can produce video
  - which `response_modalities` value would request it
  - how duration, aspect ratio or resolution are controlled
  - whether the gateway passes video parts back

**Consequence:** the Omni adapter builds a request using only the verified `/llm/generate` envelope (text and an image part). It is marked **unverified, generation disabled**, and no video-specific fields are added.
