# Veo gateway capabilities, and what they mean for an exact-person talking head

Checked on 3 October 2026 against the live gateway (`/openapi.json` v0.1.0, `GET /llm/live/info`)
and Google's `python-genai` documentation. **No generation was spent on this investigation.**

## What the gateway accepts

`POST /llm/videos` takes `{model, prompt, config, location}` with `additionalProperties: false`.
It forwards `config` to Google's `GenerateVideosConfig` "unmodified". Google's
`Models.generate_videos(model, prompt, image, video, source, config)` takes the **first-frame image
as a separate `image` argument, outside `config`**. The gateway has no field for it.

| # | Capability | Supported through this gateway? | Evidence |
|---|---|---|---|
| 1 | Image-to-video (animate the uploaded photo) | **No** | No `image` / `source` field in the request schema. The image is a `generate_videos` argument, not part of `config`. |
| 2 | First-frame conditioning | **No** | Same reason. The sibling platform (`phi-service-layer`) also recorded "`GenerateVideosConfig` rejects an input image". |
| 3 | Reference-image conditioning | **Yes** | `config.reference_images[]`, type `asset`, at most 3. Any reference pins the clip to 8 s ("supported durations are [8] for feature reference_to_video"). Used live in this project. |
| 4 | Character / subject consistency | **Partial** | It comes only from (3). Veo **generates a new face guided by the reference**. Verified on two real runs: the face is a close likeness, not the photo. |
| 5 | Native audio controls | **Yes** | `config.generate_audio` is listed by the gateway itself. Creative mode sends `true`; Strict mode now sends `false` by default. |
| 6 | Dialogue / speech instructions | **Yes, prompt only** | Veo spoke a quoted script word for word in both real runs where it was quoted (`talking-head-8s-test-001`, `20261003-070451-4badfe`). It invented speech when no script was given (`20261003-025551-a090f8`). |
| 7 | Any way to preserve the exact uploaded face | **No** | Only (3) and (4) exist. `config.last_frame` is a `GenerateVideosConfig` field, but Google documents it for interpolation together with a first frame. It was not probed, because a probe could start a paid generation. Even if accepted, only the final frame would be the photo; every frame in between is generated. |
| 8 | Existing image-animation code in the repository | **No** | The repo contains voice conversion (`chatterbox_vc`), TTS, and the Voice Studio. Nothing animates a still image. |

Other gateway endpoints:
- `/llm/images` makes still images.
- `/llm/live` is an audio-only conversational socket.

Neither animates a face.

## Consequences

**The uploaded photograph cannot be the actual face of the video.**
- With only Veo through this gateway, the best available result is a reference-guided likeness: same clothing, glasses and hair, with a regenerated face.
- Voice Studio therefore never reports a strict project as COMPLETED.
- A person may accept the output as a **likeness** (state `LIKENESS_ACCEPTED`), and every report states "exact photo animated: No".

**Lip animation is Veo's, not driven by the approved audio.** Veo animates a talking mouth for the quoted words, and Chatterbox's audio is then aligned to when that mouth moves. How well that can be checked depends on Veo audio:

| Veo audio | How timing is found | Measured on the real Vinay clip, against Veo's true audio |
|---|---|---|
| **Off** (default, as instructed) | Visible mouth movement only | **292 ms early**. Not verifiable inside the app, so it is reported as "not measurable". |
| Reference only (never in the output) | Veo's own speech regions | **0 ms** offset, r = 0.686 (Veo's native r = 0.604). Verified. |

Mouth *shapes* are never re-animated to the approved voice in either mode. Timing alignment is the most that is achievable with this stack.

## What would change this

A first-frame or image-to-video input on the gateway would make the photo the actual first frame: a gateway change that passes `image` (or `source`) to `generate_videos`. That stays within Veo and needs no new provider. Even then, the frames after the first are generated, so identity could still drift during the clip.
