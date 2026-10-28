# ZAP: gesture-triggered meme camera for Zoom

ZAP sits between your webcam and Zoom. Snap your fingers and you turn to dust like Thanos;
raise both hands and you become *Absolute Cinema*; pinch your forehead and your face morphs
into Michael Scott's. Everything runs locally on your PC: no cloud, no account.

```
webcam -> ZAP (MediaPipe hand/face tracking + effects) -> OBS Virtual Camera -> Zoom
```

## Gestures

| Do this | You get |
|---|---|
| **Snap once** | Thanos: you crumble to ash and blow away; the room stays. Snap again to come back |
| **Snap twice** | "To Be Continued" freeze-frame, 3-2-1, then ZAP leaves the meeting (cancel: open palm, Esc or a snap) |
| **Both thumbs up** | Money rain + **DEAL CLOSED** stamp |
| **Both hands up beside your head** | *Absolute Cinema* (black and white, your own face) |
| **Hand on your forehead** | Your face morphs into Michael Scott's facepalm, and back |
| **Eyebrows up** (no hands) | Your face morphs into Jim's stare at the camera |
| **Fingers at your collar** | Harvey Specter buttons his jacket, with **your face** on him |
| **Cover your face with both hands, then drop them** | Crying Michael Jordan face + sad mode |

Snaps are **seen**, not heard. The camera watches your thumb press the middle finger and
the flick, so a noisy room doesn't matter. Keep the snapping hand in view. Every gesture can be
remapped in `config.toml`, and every effect also has a hotkey.

## Controls

| Key | Action |
|---|---|
| Ctrl+Alt+A | Arm / disarm gestures (ZAP starts disarmed: plain camera) |
| Ctrl+Alt+P, or Space in the preview | **Panic**: stop every effect, show the plain camera |
| Ctrl+Alt+1..7 | Fire an effect by hand: 1 Thanos, 2 money, 3 Cinema, 4 sad MJ, 5 Michael Scott, 6 Jim, 7 Harvey |
| Ctrl+Alt+End | "To Be Continued" outro (then leave) |
| Ctrl+Alt+B | Capture a clean background: step out of view for 3 s (makes Thanos and MJ look right) |
| Ctrl+Alt+S | Sound on / off |
| Ctrl+Alt+R | Reload `config.toml` |
| Ctrl+Alt+Up / Down | Snap sensitivity |

The preview window (never sent to Zoom) shows a HUD with fps, what the gestures see and any warnings.

## Install (Windows 10/11)

1. Install [OBS Studio](https://obsproject.com) once. ZAP uses its virtual camera driver; OBS itself does not need to run.
2. Install [uv](https://docs.astral.sh/uv/), then in this folder:
   ```
   uv sync
   uv run zap fetch-models              # MediaPipe models (~12 MB)
   uv run zap fetch-models --faceswap   # optional: your face on Harvey (~450 MB, see below)
   uv run zap doctor                    # checks camera, virtual camera, models
   uv run zap run
   ```
3. Put your meme files in `%APPDATA%\ZAP\assets\memes` (they are not included; see below).
4. In Zoom: Settings, Video, Camera = **OBS Virtual Camera**.

## Zoom checklist

- Turn off Zoom's own effects: virtual background/blur, filters, studio effects, touch up, low-light, auto-framing.
- Turn off **Reactions, "Use hand gestures to activate reactions"**, or Zoom reacts to your thumbs-up too.
- Untick "Mirror my video" (ZAP already mirrors your view; captions stay readable for everyone).
- Start ZAP **before** joining the meeting.
- Leaving the meeting is in dry-run mode until you set `[zoom] dry_run = false`; until then the double snap only logs "WOULD LEAVE".

## Memes and models

Meme pictures and clips are copyrighted, so ZAP ships none. Name your files like this (png, jpg, gif or mp4):

| File | Used for |
|---|---|
| `michael_scott_forehead.*` | Michael Scott facepalm |
| `jim_stare.*` | Jim's stare at the camera |
| `harvey_button.*` | Harvey buttoning his jacket (a clip where he faces the camera works best) |
| `crying_jordan.png` | Crying Jordan face cutout (transparent PNG) |

Without a file, the effect shows a caption card instead.

**Face swap** (your face on Harvey) uses InsightFace's *inswapper* and *ArcFace* models, the same
ones FaceFusion and Deep-Live-Cam use. They are **licensed for non-commercial research use only**
and are downloaded separately by `zap fetch-models --faceswap`. ZAP learns your face from a few
frames of the live camera and never saves it. On a PC with a DirectX 12 GPU, the swap takes about
a second; on CPU only, it is ready about a minute after ZAP first sees you.

## How it works

- **Tracking**: MediaPipe gesture recognizer, face landmarker (with blendshapes for the eyebrows) and selfie segmenter, on the CPU at 640x360.
- **Snaps**: hand landmarks per frame. A "loaded" hand (thumb tip on the middle fingertip) followed within 0.45 s by the middle finger flicking toward the palm counts as one snap; snaps within 0.8 s are counted together.
- **Morph**: both faces get the same mesh of landmarks plus rings around the head; each picture is warped triangle by triangle onto the in-between shape and the two are cross-dissolved (the classic film morph).
- **Thanos**: a clean plate of the room shows through wherever the person mask says you are, while your last frame breaks into wind-blown ash.
- **Output**: pyvirtualcam into the OBS Virtual Camera at 1280x720, 30 fps.

## License

GPL-3.0 (pyvirtualcam is GPL). Meme assets and the optional face-swap models are not part of
this project and keep their own licenses.
