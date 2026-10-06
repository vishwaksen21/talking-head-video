# Voice Studio

Generate a video from a prompt through the Utilities API, then replace its speech with the
same words spoken in your own reference voice, using the Chatterbox installation in this
repository. The result is exported as an MP4.

```
prompt ─► Utilities API (Veo) ─► generated.mp4
            ─► FFmpeg: extract soundtrack ─► find where speech is
            ─► Utilities API (Gemini): transcribe each speech region
            ─► you review / edit the words            (the run waits here)
            ─► Chatterbox TTS in the reference voice, one line at a time
            ─► place each line where the original line started
            ─► FFmpeg: original pictures + cloned voice ─► final.mp4
```

## Start

From the repository root (`chatterbox-video/`):

```bash
.venv/bin/python -m studio            # http://127.0.0.1:8800
.venv/bin/python -m studio --port 9000
```

Requires the repo's existing `.venv` (Chatterbox and its Hugging Face weights are reused,
nothing is downloaded again) and FFmpeg on PATH (`brew install ffmpeg`).

Configuration: copy `.env.example` to `.env`, or enter the base URL and key in **Settings**.
The key stays on the server. The browser only learns whether a key is set.

Target voice: put a clean 6–10 s speech clip in `studio/data/voices/` (WAV/MP3/FLAC/M4A).
`reference.wav` is the clip uploaded earlier to the Chatterbox Gradio app. Only the first
10 s are used. Chatterbox's built-in voice is always available too.

## Using it

1. **Settings**: save the base URL and key, then press *Test connection*. The video models the
   API offers are listed.
2. **Create**: write a prompt that includes the spoken words in quotes, pick a model and
   options, then press *Generate video*. Each press uses one generation from your quota.
3. The project page follows the job: video → soundtrack and words → cloned voice → final.
4. **Review**: the transcript stops for you. Correct words, adjust start times, add or remove
   lines, choose the voice and what happens if the new speech runs long. Then press
   *Create cloned voice*.
5. **Final**: compare original and final side by side, read the sync report, and use
   *Download MP4*. A completed project can be re-voiced with different words without
   generating the video again.
6. **Projects**: the full history. Any project can be reopened.

## Scenes, reference images and approved scripts

- **Scenes:** up to 4 prompts per project. Each scene is one generation. Its operation id is stored as soon as the API
  accepts it, and a finished clip (`video/scene_NN.mp4`) is never requested again. Clips are joined with a 0.5 s
  crossfade. A generation beyond the scene count is refused until you press *Generate again (uses 1 generation)*.
- **Reference image:** uploaded on the Create page and sent to Veo as `config.reference_images[]` with type `asset`.
  Veo 3.1 models only, up to 3 images. Any reference pins the clip to 8 s. Each scene can opt out.
  This gateway has no image-to-video (first-frame) mode.
- **Approved script:** one sentence per line. Those exact words are spoken in the target voice, spread across the
  scenes in order. What the video itself said is transcribed and compared, and a mismatch is flagged rather than voiced.
- **End card:** optional. The last 2 s fade to white with the real `logo.png`. The model is never asked to draw a logo.
- **Export:** a completed project also gets `reference/`, `video/`, `audio/` (original Veo audio, script, each line,
  cloned voice) and `output/<export name>`.
- **Voices:** add one in *Settings → Add a voice file*. An existing voice is never overwritten.

## Video direction (structured, provider-specific)

Strict projects carry a validated **creative direction** (`studio/server/direction.py`, schema 1.0).
- **What it covers:** subject, scene, camera, performance, speech, audio, realism and output.
- **Presets:** *Natural Human* (the default), *Professional Presenter*, *Expressive*, or *Custom* overrides.
- **Validation:** unknown fields and bad values are rejected with the field named, and contradictory settings are refused.

`prompt_compiler.py` turns it into a deterministic eight-section prompt with the script quoted
exactly, plus a negative prompt. `adapters.py` builds each provider's real request:

- **Veo:** `POST /llm/videos {model, prompt, config{number_of_videos, duration_seconds, aspect_ratio, resolution,
  generate_audio, negative_prompt, reference_images[]}, location}`. Only fields verified for the gateway are sent;
  everything else is prompt text.
- **Omni** (`gemini-omni-*`): **preview only**. Only the verified `POST /llm/generate {model, contents, location}`
  envelope is built, with the prompt and the image. Video output, duration, aspect ratio, resolution and audio controls
  are not verified, and the gateway's response has no video field, so generation is disabled.

*Preview request* in the Video direction panel (`POST /api/direction/preview`) shows the direction JSON, the
provider request (image bytes redacted) and the compiled prompt. It never calls a provider. The compiled prompt and
request are saved in the project (`params.strict.direction_record`, `reports/direction.json`) when generation is approved.

## Provider-call gate

Every Utilities API call goes through `utils_api.gated_client`. That covers model listing, generation, polling and Gemini
transcription. The gate is **off by default** (`STUDIO_PROVIDER_CALLS`). While it is off:
- **Still works:** previews, photo checks, speech preparation and audits, all offline.
- **Model lists:** fall back to the verified Veo 3.1 ids.
- **Generation buttons:** disabled.
- **Restarts:** do not resume polling.

To allow the explicit generation workflow, add `STUDIO_PROVIDER_CALLS=on` to `.env` and restart.

## How synchronisation works

- Speech regions are measured from the audio (adaptive threshold, 20 ms frames), and each
  region is transcribed separately. Line start times are therefore the real onsets. In
  testing, Gemini's own timestamps were 1.7 s off.
- Each cloned line starts where its original line started, so the original pauses survive.
- A line longer than its slot (up to the next line, or the end of the video) is sped up
  with FFmpeg `atempo` (pitch kept), but by no more than the *max speed-up* (default +15%).
  Anything still too long pushes later lines back, and this is reported.
- If the speech still ends after the video, either the **last frame is held** (`tpad`
  clone mode, never black frames) until the last word ends (the default), or, with
  *Speak up to 30% faster first*, the speed-up cap is raised to +30% before holding.
  The last sentence is never cut.
- The original soundtrack is replaced, not mixed in. Speech is normalised to about −20 dBFS
  with 8 ms edge fades. When nothing is frozen, the video stream is copied bit for bit.

A video with no audio track, or silent or music-only audio, does not fail. The project waits
at the review step with an empty line for you to type what should be said.

## Job states

`IDLE → GENERATING_VIDEO → EXTRACTING_AUDIO → TRANSCRIBING → AWAITING_TRANSCRIPT_REVIEW →
SYNTHESIZING_VOICE → SYNCHRONIZING → COMPOSING → COMPLETED`, plus `FAILED` (with the stage
it failed in) and `CANCELLED`. *Retry* resumes from the failed stage. A generation that
already has an operation id resumes polling it rather than paying for a new one. On server
restart, running generations are re-polled and other interrupted jobs are marked failed
with a Retry path.

## Code

| File | Responsibility |
|---|---|
| `server/config.py` | `.env` + saved settings, base-URL normalisation, key never exposed |
| `server/utils_api.py` | Gateway client: `X-API-Key`, timeouts, retries/backoff, 429 + Retry-After, error messages |
| `server/catalog.py` | Video-model discovery from `/llm/models` and per-model option validation |
| `server/store.py` | SQLite project history and states |
| `server/media.py` | ffprobe, audio extraction, speech-region detection |
| `server/transcribe.py` | Per-region transcription via `/llm/generate` |
| `server/voice.py` | The existing Chatterbox TTS: one model instance, sentence chunking, OOM handling |
| `server/sync.py` | Placement plan, tempo changes, voice-track render |
| `server/compose.py` | Final MP4 mux and verification |
| `server/pipeline.py` | Background job manager and state machine |
| `server/main.py` | HTTP API and static frontend |
| `web/` | Frontend (plain HTML/CSS/JS, Prudential design system from `../DESIGN.md`) |

## API facts this relies on (verified live)

- `GET /llm/models?location=us-central1` lists ids only. Video models are the `veo-*` ids.
  Per-model options (lengths, shapes, quality, sound) come from Google's Veo documentation,
  because the gateway does not publish them. The gateway notes that only `veo-3.1-*` ids are
  known to be enabled, and the UI flags the others.
- `POST /llm/videos {model, prompt, config, location}` returns an operation. Poll it with
  `GET /llm/videos?operation=…`. The finished video arrives inline as base64. There is no
  cancel endpoint.
- `/healthz` cannot be used: Cloud Run intercepts it. The connection test lists models instead.

## Tests

```bash
.venv/bin/python -m pytest studio/tests -q          # isolated: mocked gateway, fake voice, real FFmpeg
STUDIO_REAL=1 STUDIO_FIXTURE_VIDEO=/path/to/clip.mp4 \
  .venv/bin/python -m pytest studio/tests/test_real_integration.py -s   # live transcription + real Chatterbox
```

The mocked tests prove the app's logic only. They are never evidence that the real API works.
