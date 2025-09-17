# Yui (Python) - Realtime Text Client

A minimal Python client to talk to OpenAI Realtime over WebSocket in text-only mode.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a .env file (auto-loaded):

```bash
echo "OPENAI_API_KEY=your_api_key" > .env
# Optional:
echo "OPENAI_REALTIME_URL=wss://api.openai.com/v1/realtime?model=gpt-realtime" >> .env
```

## Run (Text mode)

```bash
python yui.py
```

- Type a prompt and press Enter.
- Assistant responses stream to the terminal.

Options:

```bash
python yui.py --system-prompt "You are a helpful assistant"
```

## Run (Voice mode)

Install audio dependencies:

```bash
pip install -r requirements.txt
```

Run with microphone and speaker:

```bash
python yui.py --voice
```

## Notes

- Voice mode uses the OpenAI Agents SDK (`openai-agents[voice]`) with `sounddevice` for audio I/O.
