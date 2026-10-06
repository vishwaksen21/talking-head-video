# Talking-head failures: root-cause analysis

Branch `strict-talking-head`, 3 October 2026.

The failing run is project `20261003-025551-a090f8`. Its reference upload is byte-identical to
`sample/vinay.png` (SHA-256 `625e8e39…`). Its approved script was "Prudential Health India helps
people make informed choices for their health and well-being." Veo said:

> Welcome everyone. I'm excited to share some key insights with you today. Let's get started.
> Hello, this is a test of the emergency broadcast system.

## Execution path traced (before this fix)

1. **Create page (`web/app.js` → `renderCreate`).**
   - The image is uploaded to `POST /api/uploads/image` and stored at `data/uploads/<sha16>/<name>`.
   - Generate sends `{scenes:[{prompt, use_references}], reference_ids, script, …}` to `POST /api/projects`.
2. **`main.py` → `create_project`.**
   - It validates the model against the live `/llm/models` listing.
   - It builds `config` with `catalog.build_video_config`, where a reference forces 8 s.
   - It copies the image to `projects/<id>/reference/`, records `reference_files`, and stores `script` as a separate parameter.
3. **`pipeline._generate` → `_generate_scene` → `_scene_config`.**
   - For each scene with `use_references`, it reads the image bytes, base64-encodes them, and sends `config.reference_images=[{image:{image_bytes,mime_type}, reference_type:"asset"}]`.
   - **The prompt is sent exactly as typed.**
4. **`_extract_and_transcribe`.** It extracts the Veo audio, transcribes it with Gemini, and compares it with the script.
5. **`submit_transcript` → `_synthesize_and_compose`.**
   - Chatterbox speaks the approved lines, which are placed on the timeline.
   - FFmpeg muxes **the TTS track** over the Veo pictures, with the original audio dropped.

Checked against the code, tests and stored data:

| Question | Finding |
|---|---|
| Where is the uploaded image stored? | `data/uploads/<sha16>/`, then copied to `projects/<id>/reference/` |
| Is it actually sent to Veo? | **Yes.** `test_two_scenes_with_reference_and_approved_script` decodes the outgoing request and checks the bytes. The failing run's parameters show `reference_files` and `use_references: true`. |
| Is the right image tied to the right scene? | Yes. It is per scene, controlled by the checkbox. |
| Does the checkbox reach the backend? | Yes. `scenes[].use_references` is stored and read in `_scene_config`, and a test confirms scene 2 opts out. |
| Is Veo's dialogue treated as authoritative? | No. When a script exists, the approved script is what is voiced. |
| Does the approved script reach Chatterbox? | Yes, line by line. |
| Is Veo audio used as the final audio? | **No.** `compose()` maps `1:a:0`, the TTS track only. Verified on earlier outputs: the final correlates 1.000 with TTS and 0.039 with Veo. |
| Hidden fallback that invents a new person? | No code path drops the reference silently. The model itself generates a new face (see I-1). |
| Duration reconciliation? | The TTS lines were spread evenly (`spread_over_scenes`, first line at 0.4 s) **after** the video existed. Nothing checked, before spending, that the script fits the clip. |

## Root causes: identity mismatch

**I-1. Veo is not given the photo as the subject; it is given an "ingredient".**
- The gateway accepts only `model`, `prompt`, `config` and `location` (`additionalProperties: false`, re-checked live today).
- `config` is Google's `GenerateVideosConfig`. It has no first-frame or image-to-video input, so the only identity mechanism is `reference_images` with type `asset`.
- Veo uses that image as guidance and **synthesises a new face**. The failing run shows the same outfit, glasses and hair, but a different face.
- **This cannot be fixed by code or prompt wording while the only provider allowed is Veo through this gateway.**

**I-2. The reference was not prepared for identity.**
- The raw screenshot was sent: 900×980, with the face only about 232 px.
- It also carried a branded backdrop and a Google Lens icon, so clothing and background dominate what Veo is told to keep.

**I-3. Text cannot enforce identity.** The prompt's "maintain the person's facial identity" is a request, not a constraint.

**I-4. No identity validation existed.** A lookalike was presented as an ordinary result, with nothing marking it as unverified.

## Root causes: dialogue mismatch

**D-1. The approved script never reached Veo.**
- `create_project` sends the user's prompt verbatim and keeps `script` only for the later TTS step.
- With `generate_audio: true` and a prompt asking the person to be "speaking naturally… delivering a short corporate message", Veo had to invent words.
- The one earlier run where the script *was* quoted in the prompt (`talking-head-8s-test-001`) came back word for word.

**D-2. Wrong order: video first, speech second.** Veo's mouths were formed for the invented words. Replacing the audio afterwards cannot fix the lips.

**D-3. TTS output was never verified.** Nothing transcribed Chatterbox's audio and compared it with the script before use.

**D-4. The UI made Veo's dialogue prominent.** The review step plays the Veo soundtrack ("Original soundtrack"), which contains the invented speech. The final MP4 did not contain it, but that was not shown clearly side by side.

**D-5. Unverified: Veo's prompt rewriting.** `GenerateVideosConfig.enhance_prompt` may rewrite prompts. Testing it costs a generation, so it was not tested.

## What the fix does

This is implemented on this branch.

**Strict Talking Head mode.** Speech comes first, then video:
1. Validate the script and voice, and record the consent attestation.
2. Prepare the reference: Apple Vision must find exactly one usable face; it is then cropped around the face. The original file stays untouched.
3. Chatterbox speaks the **exact** script. The result is transcribed locally (Whisper base.en, already cached) and compared word by word; a mismatch blocks the job.
4. **You approve the audio** before any generation is spent.
5. One Veo request is **built by the app**: the exact script is quoted as the only dialogue, with a pace taken from the TTS duration. A script that cannot fit 8 s is refused **before** spending.
6. Veo's own speech is checked against the script. If it differs, its mouths say other words and the job needs review; it is not marked a success.
7. The TTS is split at its word gaps and fitted to Veo's measured phrases, with at most 15% speed change.
8. Validation:
   - lip-sync: mouth opening vs. audio envelope, measured from Apple Vision landmarks
   - identity: Apple Vision face geometry against the reference, an advisory check (see limitations)
   - final output: streams, duration, black frames, clipping, and a fresh transcript of the final
9. **Completed only after a person confirms the identity.** Until then the state is `NEEDS_REVIEW`.

**Fallback policy.** Strict mode never falls back to anything. It stops with the failing check named.

## Limitations that remain (stated, not hidden)

- **Exact identity:** cannot be guaranteed while Veo is the only allowed generator and the gateway offers no image-to-video. You declined local photo-animation models.
- **Identity check:** no face-recognition model is installed (by request).
  - Apple Vision's generic image feature print could not separate the Veo lookalike (mean distance 0.666) from a different person (0.684), so it is not used for identity.
  - Landmark geometry separates them only weakly (lookalike 0.19, different people 0.32 and 0.41, overlapping ranges), so it is advisory, and a person must confirm.
- **Lip-sync:** timing-level only. Veo forms its own mouth shapes for the same words; nothing re-animates lips to the TTS.

## Calibration findings during implementation (measured, not assumed)

**Lip-sync measure.**
- Vision's inner-lip landmarks proved unusable. The lip opening is only 7–10 px at 720p, and it correlated r = 0.07 even with Veo's own, perfectly synced audio.
- The validator therefore measures **mouth-region pixel motion** (Vision locates the mouth) against the voice loudness envelope.

| Audio tested against the presenter's mouth (talking-head-8s-test-001) | Lag at peak |
|---|---|
| Veo native | 0 frames (r = 0.61) |
| The same voice delayed 300 ms | −4 frames |
| Delayed 500 ms | −9 frames |
| Unrelated audio | −12 frames |

- It caught a real defect: the earlier "aligned" output was **125 ms early** (+3 frames).
- Strict mode now measures that residual offset and corrects it once (≤ 0.35 s). Rehearsal result: 0 frames.

**Identity measure.**
- The first geometry included mouth-dependent proportions. On the *same* face it varied 0.06–0.30 while the person spoke, as much as the gap between different people.
- The expression-stable set (eyes-to-nose, nose width, jaw width, eye width) gives:

| Pair | Mean distance |
|---|---|
| Same person | 0.062 (max 0.101) |
| Different people | 0.104–0.199 |
| Veo's lookalike of the reference | 0.064 |

- Thresholds: median ≤ 0.09, frame-to-frame ≤ 0.15.
- The lookalike row is why a person must always confirm identity.

**Reference preparation.** The derived reference uses Apple Vision person segmentation, which is built into macOS. It replaces the branded backdrop with plain grey and keeps the person's pixels unchanged. The original upload stays byte-identical.

**Speech length.** Chatterbox takes vary: the same script measured 6.34 s, 7.5 s, 7.2 s and 5.49 s. Takes over 7.2 s are refused before any generation, and the UI offers "Generate the speech again".

**Proper nouns.** Whisper heard "Potential" for "Prudential". The check fails and labels it "sounds alike: possibly a recognition error". Approving such a take needs an explicit override after listening; the script is never rewritten.
