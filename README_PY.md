# Yui (Python) — xAI Grok Voice Agent

Realtime voice client for xAI's Voice Agent API.

## Setup

Requires Python 3.11+ and PortAudio (for `sounddevice`).

### macOS

```bash
brew install portaudio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Raspberry Pi (Raspberry Pi OS / Debian-based)

Pi OS Bookworm ships Python 3.11. On Bullseye install Python 3.11 via `pyenv` or
upgrade the OS first.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev \
                    portaudio19-dev libatlas-base-dev \
                    libsndfile1 ffmpeg git

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes for Pi:

- A Pi 4 or Pi 5 is recommended; the wake-word ONNX model is light, but realtime audio + websockets benefits from the extra headroom.
- List audio devices with `python -c "import sounddevice as sd; print(sd.query_devices())"`. If the default isn't your USB mic / speaker, configure it via `~/.asoundrc` (ALSA) so it becomes the system default.
- If `pip install` fails on `numpy`/`scipy` wheels, make sure you're on 64-bit Pi OS — 32-bit builds wheels from source and can take a long time.

Create a `.env` file (auto-loaded):

```bash
echo "XAI_API_KEY=your_xai_api_key" >> .env
echo "GOOGLE_API_KEY=your_google_api_key" >> .env   # optional; enables weather/pollen/air-quality tools
```

`GOOGLE_API_KEY` needs the Geocoding API plus the Weather, Pollen, and Air Quality APIs enabled in Google Cloud Console.

## Run

```bash
python yui.py
```

## Options

```bash
python yui.py --voice Ara --model grok-voice-think-fast-1.0 --barge-in rms \
  --system-prompt "You are Yui, a friendly voice assistant."
```

- `--voice` — `Ara` (default), `Eve`, `Leo`, `Rex`, `Sal`
- `--model` — defaults to `grok-voice-think-fast-1.0`
- `--barge-in` — `rms` (local energy gating, default) or `server` (server-side VAD)
- `--silence-ms N` — silence after speech that ends a turn (default 750; raise if she cuts you off, lower if she feels slow)
- Press **Enter** at any time to force-interrupt the assistant.

Run `python yui.py --help` for the full flag list.

## Wake word

By default Yui starts **asleep** and only wakes when the local wake-word
detector (`hey_yoo_wee.onnx`, openWakeWord) fires. On wake, Yui greets you and
starts listening. After ~30 seconds of silence she re-arms the wake word.

```bash
python yui.py                                   # asleep until wake word
python yui.py --no-wake-word                    # always on
python yui.py --wake-word-threshold 0.4         # more sensitive (default 0.5)
python yui.py --sleep-after 60                  # 60s idle before re-arming
python yui.py --no-greet                        # silent wake (no greeting)
python yui.py --wake-word-model ./other.onnx    # different model
```

The detector runs locally — no audio is sent to xAI while asleep. Audio for
the detector is downsampled 24 kHz → 16 kHz on the fly via `scipy`.

## Memory

Yui keeps a per-user memory file at `~/.yui/memory.json` (override with
`--memory-path`). On first run she onboards by asking name + birthday and
saves them via tool calls. After onboarding she captures ongoing
observations naturally as you talk.

Three function tools handle this:

- `set_profile_field(field, value)` — structured facts (`name`, `birthday`,
  `pronouns`, `timezone`, `location`, …)
- `add_observation(text)` — free-form facts she learns in conversation
- `mark_onboarded()` — flips the first-run flag so she stops asking

Inspect or edit memory directly:

```bash
cat ~/.yui/memory.json
```

To start fresh:

```bash
rm ~/.yui/memory.json
```

The full memory snapshot is injected into the session instructions at
connect time, so Yui sees what's already known and won't re-ask.

## Notes

- Audio is PCM16 mono at 24 kHz both directions.
- WebSocket endpoint: `wss://api.x.ai/v1/realtime?model=<model>`. Available in `us-east-1` only.
- User transcripts come from `conversation.item.input_audio_transcription.completed` (configured with `grok-2-audio` in `session.update`).
- `get_weather`, `get_pollen`, and `get_air_quality` are wired up as function tools alongside the memory tools (require `GOOGLE_API_KEY`).
