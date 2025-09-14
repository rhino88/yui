# Yui - OpenAI Real-time Voice Agent

A lightweight Node + tsx script that enables real-time voice conversations with OpenAI's GPT models using live audio input/output.

## Features

- 🎤 Real-time voice recording and transcription
- 🤖 AI-powered conversation with GPT models
- 🔊 Natural text-to-speech responses
- ⚙️ Configurable voice, model, and conversation parameters
- 🧹 Automatic audio file cleanup
- 📱 Cross-platform compatibility

## Prerequisites

1. **Node.js (LTS recommended)**: Install via nvm or nodejs.org
2. **tsx (TypeScript runner)**: Installed as a dev dependency
3. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
4. **Audio Tools**: Install SoX (Sound eXchange) for audio recording/playback

### Installing SoX

**macOS:**

```bash
brew install sox
```

**Ubuntu/Debian:**

```bash
sudo apt-get install sox
```

**Windows:**
Download from [SoX website](https://sox.sourceforge.net/) or use Chocolatey:

```bash
choco install sox
```

## Setup

1. **Install dependencies:**

```bash
npm install
npm i -D tsx @types/node
```

2. **Set up environment variables:**

```bash
cp env.example .env
```

3. **Edit `.env` file and add your OpenAI API key:**

```bash
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Usage

```bash
npx tsx index.ts
```

### With Options

```bash
# Use a different voice
npx tsx index.ts --voice shimmer

# Custom system prompt
npx tsx index.ts --system-prompt "You are a coding assistant"

# Disable audio (connection test only)
npx tsx index.ts --no-audio
```

### Available Options

- `--voice <voice>`: Voice to use (alloy, ash, ballad, coral, echo, sage, shimmer, verse) (default: alloy)
- `--system-prompt <prompt>`: System prompt for the AI (default: helpful assistant named Yui)
- `--no-audio`: Disable audio input/output (connection test mode)
- `--help`: Show help message

## How It Works

1. **Realtime session**: Connects to OpenAI Realtime API over WebSocket
2. **Audio in**: Streams PCM16 microphone audio to the session
3. **AI response**: Receives text and PCM16 audio back from the model
4. **Audio out**: Plays audio through your speakers in near real-time
5. **Extras**: Server-side VAD, interruption handling, and simple backpressure

## Voice Options

- **alloy**: Neutral, balanced voice
- **echo**: Warm, friendly voice
- **fable**: Young, energetic voice
- **onyx**: Deep, authoritative voice
- **nova**: Bright, enthusiastic voice
- **shimmer**: Soft, gentle voice

## Troubleshooting

### Audio Issues

- Ensure your microphone is properly connected and configured
- Check that SoX is installed and working: `sox --version`
- Test audio recording: `sox -d test.wav trim 0 3`

### API Issues

- Verify your OpenAI API key is correct and has sufficient credits
- Check that your API key has access to the required models (GPT-4o, Whisper, TTS)

### Permission Issues

- On macOS, ensure the terminal has microphone permissions
- On Linux, check audio device permissions

## Development

Run in development mode with auto-restart:

```bash
npx tsx --watch index.ts
```

## License

MIT License - feel free to use and modify as needed.
