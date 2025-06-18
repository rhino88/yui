# Yui - OpenAI Real-time Voice Agent

A powerful Bun script that enables real-time voice conversations with OpenAI's GPT models using speech-to-text and text-to-speech capabilities.

## Features

- 🎤 Real-time voice recording and transcription
- 🤖 AI-powered conversation with GPT models
- 🔊 Natural text-to-speech responses
- ⚙️ Configurable voice, model, and conversation parameters
- 🧹 Automatic audio file cleanup
- 📱 Cross-platform compatibility

## Prerequisites

1. **Bun Runtime**: Install Bun from [bun.sh](https://bun.sh)
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Audio Tools**: Install SoX (Sound eXchange) for audio recording/playback

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

1. **Clone and install dependencies:**

```bash
bun install
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
bun run index.ts
```

### With Custom Options

```bash
# Use a different voice
bun run index.ts --voice nova

# Use a different model
bun run index.ts --model gpt-4

# Custom system prompt
bun run index.ts --system-prompt "You are a coding assistant"

# Adjust creativity
bun run index.ts --temperature 0.9

# Limit response length
bun run index.ts --max-tokens 100
```

### Available Options

- `--model <model>`: OpenAI model to use (default: gpt-4o)
- `--voice <voice>`: Voice to use (alloy, echo, fable, onyx, nova, shimmer) (default: alloy)
- `--system-prompt <prompt>`: System prompt for the AI (default: helpful assistant)
- `--temperature <number>`: Response creativity (0.0-2.0) (default: 0.7)
- `--max-tokens <number>`: Maximum response length (default: 150)
- `--help`: Show help message

## How It Works

1. **Recording**: The script records audio from your microphone for up to 10 seconds
2. **Transcription**: Audio is sent to OpenAI's Whisper model for speech-to-text conversion
3. **AI Response**: The transcribed text is sent to GPT for response generation
4. **Speech Synthesis**: The AI response is converted to speech using OpenAI's TTS
5. **Playback**: The synthesized speech is played through your speakers
6. **Cleanup**: Temporary audio files are automatically deleted

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
bun run dev
```

## License

MIT License - feel free to use and modify as needed.
