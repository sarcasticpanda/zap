# ZAP: gesture-triggered meme camera for Zoom

ZAP sits between your webcam and Zoom. Snap your fingers and you turn to dust like Thanos;
raise both hands and you become *Absolute Cinema*; give a thumbs down and your face morphs
into Michael Scott's. Everything runs locally on your PC: no cloud, no account.

```
webcam -> ZAP (MediaPipe hand/face tracking + effects) -> OBS Virtual Camera -> Zoom
```

## Gestures

Every effect has its own sign. Hold a one-hand sign **still for about a second, beside your face
(not on it)**, with the whole hand in the picture.

| Do this | You get |
|---|---|
| **Snap once** | Thanos: you crumble to ash and blow away (instantly, no waiting); the room stays. Snap again to come back |
| **Peace sign ✌️**, or **snap twice** (within 2 s) | "To Be Continued" freeze-frame, 3-2-1, then ZAP leaves the meeting (cancel: Esc, Space or a snap). Ignored for the first minute after arming |
| **One finger up ☝️** ("one sec") | **We'll Be Right Back**: the picture freezes, punches in on your face, caption and bumper music. If you walk off, the card stays up until you're back in view |
| **Thumbs up 👍** (either hand) | Money rain + **DEAL CLOSED** stamp |
| **Thumbs down 👎** | Your face morphs into Michael Scott's facepalm, and back |
| **Rock on 🤟 or 🤘** (index and pinky up) | Your face turns into Jim's, then you're in his shot, staring at the camera |
| **Both open hands up beside your head** | *Absolute Cinema* (black and white, your own face) |
| **Fingers at your collar** | Harvey Specter steps out of his car, with **your head** cut out and stuck on him, meme-style |
| **Cover your face with both hands, then drop them** | Crying Michael Jordan face + sad mode |

Snaps are **seen**, not heard. The camera watches your thumb press the middle finger and
the flick, so a noisy room doesn't matter. Keep the snapping hand in view.

The preview's gesture list shows each sign, a bar that fills while you hold it, and why it isn't
counting (*on your face*, *hold still*, *fingers?*). Every gesture can be remapped in `config.toml`
under `[bindings]`, and every effect also has a hotkey. Two older gestures are off by default because
they fired by accident in meetings: **hand on forehead** and **eyebrows up** (turn them back on with
`forehead = "cutaway:michael_scott"` / `eyebrows_up = "cutaway:jim"`; calibrate the eyebrows first with Ctrl+Alt+C).

## Controls

| Key | Action |
|---|---|
| Ctrl+Alt+A | Arm / disarm gestures (ZAP starts disarmed: plain camera) |
| Ctrl+Alt+P, or Space in the preview | **Panic**: stop every effect, show the plain camera (gestures pause for 3 s) |
| Ctrl+Alt+1..8 | Fire an effect by hand: 1 Thanos, 2 money, 3 Cinema, 4 sad MJ, 5 Michael Scott, 6 Jim, 7 Harvey, 8 We'll Be Right Back (press 8 again to come back early) |
| Ctrl+Alt+End | "To Be Continued" outro (then leave) |
| Ctrl+Alt+B | Capture a clean background: step out of view for 3 s (makes Thanos and MJ look right) |
| Ctrl+Alt+C | Calibrate the eyebrow gesture to your face (only if you turned it on): relax 3 s, then raise your brows 3 s |
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
   uv run zap fetch-models --enhance    # optional: sharper memes (36 MB, needs a DirectX 12 GPU)
   uv run zap fetch-models --faceswap   # optional: AI face swap instead of the head cutout (~450 MB)
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
| `jim_stare.*` | Jim turning to the camera, eyebrows up |
| `harvey_car.*` | Harvey stepping out of his car (any clip where his face is visible works) |
| `crying_jordan.png` | Crying Jordan face cutout (transparent PNG) |

Without a file, the effect shows a caption card instead.

Harvey gets **your head as a cutout** by default. For an AI face swap instead (`face_swap = "ai"` under
`[cutaways.harvey]`), ZAP uses InsightFace's *inswapper* and *ArcFace* models, the same
ones FaceFusion and Deep-Live-Cam use. They are **licensed for non-commercial research use only**
and are downloaded separately by `zap fetch-models --faceswap`. ZAP learns your face from a few
frames of the live camera and never saves it. On a PC with a DirectX 12 GPU, the swap takes about
a second; on CPU only, it is ready about a minute after ZAP first sees you.

## Sounds

Put sound files in `%APPDATA%\ZAP\assets\sfx`, named after the effect: `cutaway_jim`, `cutaway_michael_scott`,
`cutaway_harvey`, `money`, `sad`, `cinema`, `outro`, `presence` (vanish), `reappear`, `brb` (.wav, .mp3, .ogg or
.flac). Keep them about as long as the effect (~5 s). Missing sounds use a simple built-in placeholder. Ctrl+Alt+S
mutes everything.

`brb` is the We'll Be Right Back music (the meme uses "Oregon Spirit" by Rolf Kruger; bring your own clip). The card
stays up exactly as long as the file plays (2-12 s), or set `seconds` under `[brb]`.

## How it works

- **Tracking**: MediaPipe gesture recognizer, face landmarker (with blendshapes for the eyebrows) and selfie segmenter, on the CPU at 640x360.
- **Signs**: a sign counts when MediaPipe's gesture classifier names it *and* the finger landmarks have its shape (🤘, which MediaPipe has no class for, goes by shape), the hand is still, and no fingertip is on your face, so an index finger at your lips ("shh") or along your cheek is not "one sec". Checked against photos of 26 hand gestures and recordings of a real user: each sign is recognised and none is confused with another gesture.
- **Snaps**: hand landmarks per frame. A "loaded" hand (thumb tip on the middle fingertip) followed within 0.45 s by the middle finger curling into the palm counts as one snap (two hands pinching at once, like grabbing your collar, does not). Snaps up to 2 s apart count together; lower your hand to finish sooner.
- **Transitions**: a gentle zoom dissolve: the camera eases in on your face while the meme, framed on its face at the same size, fades in and then settles out. Small meme GIFs are AI-upscaled (Real-ESRGAN, on the GPU) once and cached.
- **Thanos**: a clean plate of the room shows through wherever the person mask says you are, while your last frame breaks into wind-blown ash.
- **Output**: pyvirtualcam into the OBS Virtual Camera at 1280x720, 30 fps.

## License

GPL-3.0 (pyvirtualcam is GPL). Meme assets and the optional face-swap models are not part of
this project and keep their own licenses.
