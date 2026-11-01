# ZAP: gesture-triggered meme camera for video calls

ZAP sits between your webcam and your call app: Zoom, Teams, Google Meet, WhatsApp, Discord,
Telegram and more. Snap your fingers and you turn to dust like Thanos;
raise both hands and you become *Absolute Cinema*; put a hand on your forehead and your face
morphs into Michael Scott's. Everything runs locally on your PC: no cloud, no account.

```
webcam -> ZAP (MediaPipe hand/face tracking + effects) -> OBS Virtual Camera -> your call app
```

## Gestures

One gesture per effect. **Calibrate once first: Ctrl+Alt+C** (about 2 minutes, see below).

| Do this | You get |
|---|---|
| **Snap once** | Thanos: you crumble to ash and blow away (instantly); the room stays. Snap again to come back |
| **Snap twice** (within 2 s) | You vanish, then ZAP leaves the call after 3 s (snap or Esc cancels and brings you back). Ignored for the first minute after arming. See [Your call app](#your-call-app) |
| **Peace sign ✌️** | **We'll Be Right Back**: the picture freezes, punches in on your face, caption and music. If you walk off, it stays up until you're back |
| **L sign** (index up, thumb out) | **To Be Continued**: freeze, sepia, the arrow slides in, music |
| **Open palm ✋** (while one of those two shows) | Ends it early |
| **Thumbs up 👍** | Money rain + **DEAL CLOSED** stamp |
| **Both open hands up beside your head** | *Absolute Cinema* (black and white, your own face) |
| **Fingers at your collar** | Harvey Specter steps out of his car, with **your head** cut out and stuck on him |
| **Cover your face with both hands, then drop them** | Crying Michael Jordan face + sad mode |
| **Hand on your forehead** | Your face morphs into Michael Scott's facepalm, and back |
| **Eyebrows up** | Your face turns into Jim's, staring at the camera |

Hold a hand sign **still for about a second, beside your face (not on it)**, with the whole hand in
the picture. Snaps are **seen**, not heard, so a noisy room doesn't matter.

The preview lists every gesture with a bar that fills while you hold it, and says why a sign isn't
counting (*on your face*, *hold still*, *fingers?*). Gestures can be remapped in `config.toml` under
`[bindings]`, and every effect also has a hotkey.

### Calibration (Ctrl+Alt+C)

Do this once, in your usual seat and light, with your glasses if you wear them. The preview walks you
through it: sit relaxed, talk, look down and back, raise your eyebrows twice, hand on your forehead,
touch your face naturally, then show each gesture once. ZAP then tunes Jim, Michael Scott, the peace
sign and the L sign to you, so talking, head tilts and touching your face don't set them off, and
tells you which gestures it saw (and what to fix for any it didn't). Esc cancels; run it again any time.

## Controls

| Key | Action |
|---|---|
| Ctrl+Alt+A | Arm / disarm gestures (ZAP starts disarmed: plain camera) |
| Ctrl+Alt+P, or Space in the preview | **Panic**: stop every effect, show the plain camera (gestures pause for 3 s) |
| Ctrl+Alt+1..9 | Fire an effect by hand: 1 Thanos, 2 money, 3 Cinema, 4 sad MJ, 5 Michael Scott, 6 Jim, 7 Harvey, 8 We'll Be Right Back, 9 To Be Continued (8 or 9 again ends it) |
| Ctrl+Alt+End | Vanish, then leave the call (same as snapping twice) |
| Ctrl+Alt+B | Capture a clean background: step out of view for 3 s (makes Thanos and MJ look right) |
| Ctrl+Alt+C | Calibrate every gesture to you (about 2 minutes; follow the preview) |
| Ctrl+Alt+S | Sound on / off |
| Ctrl+Alt+R | Reload `config.toml` |
| Ctrl+Alt+Up / Down | Snap sensitivity |

The preview window (never sent to the call) shows a HUD with fps, what the gestures see and any warnings.

## Install

ZAP runs on **Windows 10/11**, **macOS 14+ on Apple silicon** and **Linux** (Ubuntu 22.04+, Fedora; x86_64 or ARM).
First the virtual camera for your system, then the same steps everywhere.

**Windows:** install [OBS Studio](https://obsproject.com) once. ZAP uses its virtual camera driver; OBS itself does
not need to run.

**macOS:** install OBS Studio 30 or newer, open it, click *Start Virtual Camera* once and allow its camera extension
when macOS asks, then quit OBS. When ZAP first starts, allow Camera, Microphone and Accessibility for your terminal
in System Settings > Privacy & Security. (Intel Macs aren't supported: MediaPipe has no build for them.)

**Linux:** install the loopback camera and two libraries, then load the camera (again after each reboot, or add it
to `/etc/modules-load.d`). In your call app it is called **ZAP Camera**:
```
sudo apt install v4l2loopback-dkms libportaudio2 libgl1      # Fedora: akmod-v4l2loopback (RPM Fusion), portaudio
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="ZAP Camera"
```

Then install [uv](https://docs.astral.sh/uv/), and in this folder:
```
uv sync
uv run zap fetch-models              # MediaPipe models (~12 MB)
uv run zap fetch-models --enhance    # optional: sharper memes (36 MB, needs a GPU: DirectX 12, Apple, or NVIDIA)
uv run zap fetch-models --faceswap   # optional: AI face swap instead of the head cutout (~450 MB)
uv run zap doctor                    # checks camera, virtual camera, models; setup steps for your call apps
uv run zap run
```
Put your meme files in `assets/memes` in ZAP's folder (`%APPDATA%\ZAP` on Windows, `~/Library/Application
Support/ZAP` on macOS, `~/.config/ZAP` on Linux; they are not included, see below), and choose ZAP's camera in your
call app (see below).

Not on macOS and Linux yet: leaving the call by gesture, and playing meme sounds into the call (they play on your
speakers). On a Linux Wayland desktop the global hotkeys only work in X11 apps; the preview window's keys always do.

## Your call app

ZAP works in any call app that can use **OBS Virtual Camera**: Zoom, Microsoft Teams (and Teams free,
which replaced Skype), Google Meet and every other call in Chrome, Edge or Firefox, WhatsApp, Discord,
Telegram, Signal, Webex and Slack. In every app:

- Start ZAP **before** the app turns your camera on (picture black? turn the camera off and on in the call).
- Turn off the app's own background blur/replace, filters and auto-framing: they cut into the memes.
- Your own preview may look mirrored; everyone else sees ZAP the right way round.
- Meme sounds into the call: install VB-CABLE, set `[sound] output = "meeting"`, pick **CABLE Output** as the
  call's microphone, and turn the app's noise suppression down (or it removes the music).

| App | Snap twice leaves by | Also turn off |
|---|---|---|
| Zoom | Alt+Q, then "Leave meeting" (in any of Zoom's languages) | Reactions: **"Use hand gestures to activate reactions"** (it reacts to thumbs up, peace...) |
| Microsoft Teams | Ctrl+Shift+H (never "end for everyone") | Background effects. If its preview shows the OBS logo or a frozen picture, restart Teams with ZAP running, or use Teams in the browser |
| Google Meet | closing the Meet tab | "Turn off visual effects", Framing, Reactions > Hand raise gesture |
| WhatsApp | its end-call button | If the app doesn't list the camera, call from web.whatsapp.com in Chrome or Edge |
| Discord | your **Disconnect from Voice Channel** keybind (bind one in Discord and put it in `[call] discord_keys`), else the Disconnect button | Video background; noise suppression (for meme sounds) |
| Telegram | its "End Call" button | |
| Signal | Alt+Shift+E | (Signal sends 4:3: memes at the far left and right may be cut off) |
| Webex | Ctrl+L, then "Leave meeting" | Reactions: "Recognize hand gestures" |
| Slack huddles | the huddle's Leave button, only with `[call] app = "slack"` | |
| Other browser calls (Jitsi, Zoom or Discord on the web, FaceTime links, Facebook) | closing that tab | |

ZAP never presses Enter or an "end for everyone" button, and if it finds no call, or more than one, it does
nothing and tells you. Leaving is in dry-run mode until you set `[call] dry_run = false` (until then snapping
twice only logs "WOULD LEAVE"); `[call] app` picks one app instead of `"auto"`. `zap doctor` prints the steps
for the call apps on your PC, and `zap doctor --calls` shows what ZAP sees in them. Not supported: Viber (it
doesn't list virtual cameras). On macOS and Linux, leaving by gesture isn't available yet.

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

Put sound files in `assets/sfx` in ZAP's folder, named after the effect: `cutaway_jim`, `cutaway_michael_scott`,
`cutaway_harvey`, `money`, `sad`, `cinema`, `presence` (vanish), `reappear`, `brb`, `tbc` (.wav, .mp3, .ogg or
.flac). Keep them about as long as the effect (~5 s). Missing sounds use a simple built-in placeholder. Ctrl+Alt+S
mutes everything.

`brb` is the We'll Be Right Back music (the meme uses "Oregon Spirit" by Rolf Kruger) and `tbc` the To Be Continued
music (the meme uses "Roundabout" by Yes); bring your own clips. Each card stays up as long as its file plays
(2-12 s), or set `seconds` under `[brb]` / `[tbc]`.

## How it works

- **Tracking**: MediaPipe gesture recognizer, face landmarker (with blendshapes for the eyebrows) and selfie segmenter, on the CPU at 640x360.
- **Signs**: a sign counts when MediaPipe's gesture classifier names it *and* the finger landmarks have its shape, or, while the classifier sees no sign at all, a stricter shape alone (MediaPipe has no class for the L, and often misses a peace sign shown with the back of the hand). The hand must be still, and no fingertip may be on your face, so a finger at your lips or along your cheek doesn't count. Checked on photos of 26 hand gestures, real people in their rooms, and recordings of a real user: peace 10/12 and L 11/12 of the photo variants, and no other gesture (not even a hand about to snap) is taken for either.
- **Calibration**: the Ctrl+Alt+C steps are kept in memory as landmarks only (no video), replayed through the same detector with candidate settings, and the first setting that fires in its own step and never elsewhere is saved to `calibration/gestures.json`.
- **Snaps**: hand landmarks per frame. A "loaded" hand (thumb tip on the middle fingertip) followed within 0.45 s by the middle finger curling into the palm counts as one snap (two hands pinching at once, like grabbing your collar, does not). Snaps up to 2 s apart count together; lower your hand to finish sooner.
- **Transitions**: a gentle zoom dissolve: the camera eases in on your face while the meme, framed on its face at the same size, fades in and then settles out. Small meme GIFs are AI-upscaled (Real-ESRGAN, on the GPU) once and cached.
- **Thanos**: a clean plate of the room shows through wherever the person mask says you are, while your last frame breaks into wind-blown ash.
- **Output**: pyvirtualcam into the OBS Virtual Camera at 1280x720, 30 fps.

## License

GPL-3.0 (pyvirtualcam is GPL). Meme assets and the optional face-swap models are not part of
this project and keep their own licenses.
