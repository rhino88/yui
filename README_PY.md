# Yui (Python) — xAI Grok Voice Agent

Realtime voice client for xAI's Voice Agent API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (auto-loaded):

```bash
echo "XAI_API_KEY=your_xai_api_key" > .env
```

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
- Press **Enter** at any time to force-interrupt the assistant.

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
- `get_weather` is wired up as a sample function tool alongside the memory tools.
