#!/usr/bin/env bun

import {
  RealtimeAgent,
  RealtimeSession,
  OpenAIRealtimeWebSocket,
} from "@openai/agents-realtime";

// Audio libraries - these are now installed as dependencies
let mic: any = null;
let Speaker: any = null;
let recorder: any = null;

// Try to import audio libraries
try {
  const micModule = require("mic");
  const speakerModule = require("speaker");
  const recorderModule = require("node-record-lpcm16");

  mic = micModule;
  Speaker = speakerModule;
  recorder = recorderModule;

  console.log("✅ Audio libraries loaded successfully");
} catch (error) {
  console.warn("⚠️  Audio libraries failed to load:", error);
  console.warn("   Make sure you have: bun add mic speaker node-record-lpcm16");
}

interface VoiceAgentConfig {
  voice: string;
  systemPrompt: string;
  enableAudio: boolean;
}

class RealtimeVoiceAgent {
  private agent: RealtimeAgent;
  private session: RealtimeSession | null = null;
  private config: VoiceAgentConfig;
  private isConnected = false;
  private micInstance: any = null;
  private speaker: any = null;
  private audioQueue: Buffer[] = [];
  private isProcessingAudio = false;
  private audioPreBuffer: Buffer[] = []; // Pre-buffer to prevent underflow
  private minBufferSize = 3; // Minimum buffers before starting playback
  private uncaughtExceptionHandler: ((error: Error) => void) | null = null;
  private audioFailureCount = 0; // Track consecutive audio failures
  private maxAudioFailures = 10; // Disable audio after this many failures

  constructor(config: Partial<VoiceAgentConfig> = {}) {
    this.config = {
      voice: config.voice || "alloy",
      systemPrompt:
        config.systemPrompt ||
        "You are a helpful AI assistant named Yui (pronounced yoo_wee). Keep your responses concise and natural for voice conversation. Respond in English. You should start the conversation with a greeting, before waiting for the user's response.",
      enableAudio: config.enableAudio !== false, // Default to true
    };

    // Create the realtime agent
    this.agent = new RealtimeAgent({
      name: "Voice Assistant",
      instructions: this.config.systemPrompt,
      voice: this.config.voice,
    });
  }

  private async processAudioQueue() {
    if (
      this.isProcessingAudio ||
      this.audioQueue.length === 0 ||
      !this.speaker
    ) {
      return;
    }

    // Wait for minimum buffer before starting playback (prevents underflow)
    if (
      this.audioQueue.length < this.minBufferSize &&
      this.audioPreBuffer.length === 0
    ) {
      console.log(
        `🔄 Buffering audio... (${this.audioQueue.length}/${this.minBufferSize})`
      );
      return;
    }

    this.isProcessingAudio = true;

    try {
      while (this.audioQueue.length > 0) {
        const buffer = this.audioQueue.shift();
        if (buffer && buffer.length > 0) {
          // Use a simpler approach without timeout - let Speaker handle it
          try {
            await new Promise<void>((resolve, reject) => {
              this.speaker.write(buffer, (error: any) => {
                if (error) {
                  console.error("❌ Audio write error:", error);
                  this.audioFailureCount++;
                  reject(error);
                } else {
                  // Reset failure count on successful write
                  this.audioFailureCount = 0;
                  resolve();
                }
              });
            });
          } catch (writeError: any) {
            // Check if we should disable audio due to too many failures
            if (this.audioFailureCount >= this.maxAudioFailures) {
              console.error(
                "❌ Too many audio failures - disabling audio output"
              );
              this.config.enableAudio = false;
              this.audioQueue = [];
              return; // Stop processing
            }

            console.warn("⚠️  Audio write failed - skipping buffer");
            // Skip this buffer and continue with next
            continue;
          }

          // Smaller delay to maintain smooth audio flow
          await new Promise((resolve) => setTimeout(resolve, 2));
        }
      }
    } catch (error) {
      console.error("❌ Audio processing error:", error);
      // More conservative queue management
      if (this.audioQueue.length > 20) {
        console.log("🔄 Reducing audio queue size due to errors");
        this.audioQueue = this.audioQueue.slice(-10); // Keep only last 10 buffers
      }
    } finally {
      this.isProcessingAudio = false;

      // Continue processing if there's more data
      if (this.audioQueue.length > 0) {
        // Add small delay before retrying to prevent tight error loops
        setTimeout(() => {
          if (!this.isProcessingAudio) {
            this.processAudioQueue();
          }
        }, 100);
      }
    }
  }

  private setupAudio() {
    if (!this.config.enableAudio) {
      console.log("🔇 Audio explicitly disabled");
      return;
    }

    if (!Speaker || !mic) {
      console.log("🔇 Audio libraries not available - running in text mode");
      return;
    }

    try {
      // Set up speaker for audio output with better configuration
      this.speaker = new Speaker({
        channels: 1,
        bitDepth: 16,
        sampleRate: 24000,
        highWaterMark: 16384, // Even larger buffer size
        lowWaterMark: 4096, // Higher low water mark
      });

      // Handle speaker events
      this.speaker.on("error", (err: Error) => {
        console.error("❌ Speaker error:", err);
        // Try to recover by clearing the queue
        this.audioQueue = [];
        this.audioPreBuffer = [];
        this.isProcessingAudio = false;
      });

      this.speaker.on("drain", () => {
        // Speaker is ready for more data - process queue immediately
        if (this.audioQueue.length > 0 && !this.isProcessingAudio) {
          setImmediate(() => this.processAudioQueue());
        }
      });

      // Add 'close' event handler
      this.speaker.on("close", () => {
        console.log("🔊 Speaker closed");
      });

      // Preload speaker with larger silence buffer to prevent initial underflow
      const silenceBuffer = Buffer.alloc(8192, 0); // Larger silence buffer
      this.speaker.write(silenceBuffer);

      // Create additional silence buffers in pre-buffer
      for (let i = 0; i < this.minBufferSize; i++) {
        this.audioPreBuffer.push(Buffer.alloc(2048, 0));
      }

      // Set up microphone for audio input
      this.micInstance = mic({
        rate: "24000",
        channels: "1",
        debug: false,
        exitOnSilence: 6,
        device: "default",
      });

      const micInputStream = this.micInstance.getAudioStream();

      micInputStream.on("data", (data: Buffer) => {
        if (this.session && this.isConnected && data.length > 0) {
          // Convert Buffer to ArrayBuffer for the realtime API
          const arrayBuffer = new ArrayBuffer(data.length);
          const view = new Uint8Array(arrayBuffer);
          view.set(data);
          this.session.sendAudio(arrayBuffer);
        }
      });

      micInputStream.on("error", (err: Error) => {
        console.error("❌ Microphone error:", err);
      });

      micInputStream.on("silence", () => {
        console.log("🔇 Silence detected");
      });

      console.log("🎤 Audio input/output initialized successfully");
      return true;
    } catch (error) {
      console.error("❌ Failed to setup audio:", error);
      this.config.enableAudio = false;
      return false;
    }
  }

  async start() {
    console.log("🚀 Starting OpenAI Realtime Voice Agent...");

    if (!process.env.OPENAI_API_KEY) {
      console.error("❌ OPENAI_API_KEY environment variable is required");
      console.log("Please set it in your .env file or export it in your shell");
      process.exit(1);
    }

    try {
      // Create WebSocket transport for terminal usage
      const transport = new OpenAIRealtimeWebSocket();

      // Create a new realtime session with better configuration
      this.session = new RealtimeSession(this.agent, {
        transport: transport,
        model: "gpt-4o-realtime-preview-2025-06-03",
      });

      // Wrap all event handlers in try-catch to prevent SDK errors from crashing
      const safeEventHandler = (
        eventName: string,
        handler: (event: any) => void
      ) => {
        this.session!.on(eventName as any, (event: any) => {
          try {
            handler(event);
          } catch (error: any) {
            console.warn(
              `⚠️  Error in ${eventName} handler:`,
              error?.message || error
            );
            // Don't crash on SDK errors
          }
        });
      };

      // Set up audio event handling with error protection
      safeEventHandler("audio", (event: any) => {
        if (
          this.speaker &&
          this.config.enableAudio &&
          event.data &&
          event.data.byteLength > 0
        ) {
          const buffer = Buffer.from(event.data);

          // Prevent queue from growing too large (which can cause memory issues)
          if (this.audioQueue.length > 25) {
            console.log("⚠️  Audio queue too large, dropping oldest audio");
            this.audioQueue.shift();
          }

          this.audioQueue.push(buffer);

          // Only log every 5th audio packet to reduce noise
          if (this.audioQueue.length % 5 === 0) {
            console.log(`🔊 Audio buffered (queue: ${this.audioQueue.length})`);
          }

          // Process audio queue immediately if not already processing
          if (!this.isProcessingAudio) {
            setImmediate(() => this.processAudioQueue());
          }
        }
      });

      // Set up conversation event handlers with error protection
      safeEventHandler("conversation.item.created", (event: any) => {
        if (event?.item?.type === "message" && event?.item?.role === "user") {
          console.log(`👤 User: ${event.item.content?.[0]?.text || "[audio]"}`);
        }
      });

      safeEventHandler("conversation.item.completed", (event: any) => {
        if (
          event?.item?.type === "message" &&
          event?.item?.role === "assistant"
        ) {
          console.log(
            `🤖 Assistant: ${
              event.item.content?.[0]?.text || "[audio response]"
            }`
          );
        }
      });

      safeEventHandler("response.created", (event: any) => {
        console.log(`🎯 Response started`);
      });

      safeEventHandler("response.done", (event: any) => {
        console.log(`✅ Response completed`);
      });

      // Handle errors more gracefully
      this.session.on("error", (error: any) => {
        if (error?.error?.code === "response_cancel_not_active") {
          // Completely ignore this error - it's not critical and happens when
          // the SDK tries to cancel responses that are already finished
          // Don't log anything - this is normal behavior
          return;
        } else if (error?.error?.type === "invalid_request_error") {
          console.warn(
            "⚠️  Invalid request error:",
            error?.error?.message || error
          );
        } else if (error?.type === "connection_error") {
          console.error("🔌 Connection error:", error?.message || error);
          console.log("💡 Check your internet connection and API key");
        } else {
          console.error("❌ Session error:", error);
        }
      });

      // Add more detailed event logging to understand the flow
      safeEventHandler("response.audio_transcript.delta", (event: any) => {
        if (event?.delta) {
          process.stdout.write(event.delta);
        }
      });

      safeEventHandler("conversation.interrupted", () => {
        console.log("\n🚫 Conversation interrupted");
      });

      // Add global error handler for uncaught SDK errors
      this.uncaughtExceptionHandler = (error: Error) => {
        if (
          error.message.includes("undefined is not an object") &&
          error.stack?.includes("realtimeSession.mjs")
        ) {
          console.warn("⚠️  SDK error caught and handled:", error.message);
          console.log(
            "💡 This is a known SDK issue with malformed responses - continuing..."
          );
          // Don't crash on SDK bugs
          return;
        }
        // Re-throw other uncaught exceptions
        throw error;
      };

      process.on("uncaughtException", this.uncaughtExceptionHandler);

      // Connect to the realtime API
      await this.session.connect({
        apiKey: process.env.OPENAI_API_KEY,
      });

      this.isConnected = true;
      console.log("✅ Connected to OpenAI Realtime API via WebSocket");
      console.log(`Voice: ${this.config.voice}`);
      console.log("Instructions:", this.config.systemPrompt);

      // Setup audio after connection
      const audioSetup = this.setupAudio();

      if (this.config.enableAudio && audioSetup && this.micInstance) {
        console.log("\n🎤 Starting microphone...");
        this.micInstance.start();
        console.log("🔴 Recording started - speak now!");
        console.log("💡 The AI will respond with voice when you speak");
        console.log("💡 Say something like 'Hello, how are you?' to test");
      } else {
        console.log(
          "\n💬 Audio not available - connection established for testing"
        );
        console.log("💡 You can still test the connection, but no audio I/O");
      }

      console.log("\n🛑 Press Ctrl+C to stop the agent.\n");
    } catch (error) {
      console.error("❌ Failed to connect to OpenAI Realtime API:", error);
      console.log("\n💡 Make sure you have:");
      console.log("  1. A valid OPENAI_API_KEY in your environment");
      console.log("  2. Access to the OpenAI Realtime API");
      console.log("  3. Sufficient credits in your OpenAI account");
      process.exit(1);
    }
  }

  async stop() {
    console.log("🛑 Stopping voice agent...");

    // Stop microphone first
    if (this.micInstance) {
      try {
        this.micInstance.stop();
        console.log("🎤 Microphone stopped");
      } catch (error) {
        console.warn("⚠️  Error stopping microphone:", error);
      }
    }

    // Clean up audio processing
    this.isProcessingAudio = false;
    this.audioQueue = [];
    this.audioPreBuffer = [];

    // Remove global error handler
    if (this.uncaughtExceptionHandler) {
      process.removeListener(
        "uncaughtException",
        this.uncaughtExceptionHandler
      );
      this.uncaughtExceptionHandler = null;
    }

    // Stop speaker
    if (this.speaker) {
      try {
        // Write any remaining buffered audio with a small delay
        if (this.audioQueue.length > 0) {
          console.log("🔊 Flushing remaining audio...");
          await new Promise((resolve) => setTimeout(resolve, 100));
        }

        this.speaker.end();
        console.log("🔊 Speaker stopped");
      } catch (error) {
        console.warn("⚠️  Error stopping speaker:", error);
      }
    }

    // Disconnect session
    if (this.session && this.isConnected) {
      try {
        // Note: RealtimeSession doesn't have a disconnect method
        // Setting isConnected to false and the session will clean up automatically
        this.isConnected = false;
        console.log("🔌 Session marked as disconnected");
      } catch (error) {
        console.warn("⚠️  Error with session cleanup:", error);
        this.isConnected = false;
      }
    }

    console.log("✅ Voice agent stopped");
  }

  getStatus() {
    return {
      connected: this.isConnected,
      voice: this.config.voice,
      instructions: this.config.systemPrompt,
      audioEnabled: this.config.enableAudio,
      audioQueueSize: this.audioQueue.length,
    };
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);

  // Parse command line arguments
  const config: Partial<VoiceAgentConfig> = {};

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--voice":
        config.voice = args[++i];
        break;
      case "--system-prompt":
        config.systemPrompt = args[++i];
        break;
      case "--no-audio":
        config.enableAudio = false;
        break;
      case "--help":
        showHelp();
        return;
      default:
        console.error(`❌ Unknown argument: ${args[i]}`);
        showHelp();
        return;
    }
  }

  const agent = new RealtimeVoiceAgent(config);

  // Handle graceful shutdown
  process.on("SIGINT", async () => {
    console.log("\n👋 Shutting down gracefully...");
    await agent.stop();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    console.log("\n👋 Received SIGTERM, shutting down...");
    await agent.stop();
    process.exit(0);
  });

  await agent.start();

  // Keep the process alive and show status
  setInterval(() => {
    const status = agent.getStatus();
    if (status.connected) {
      console.log(
        `🔄 Voice agent running... Audio: ${
          status.audioEnabled ? "ON" : "OFF"
        } Queue: ${status.audioQueueSize}`
      );
    }
  }, 60000); // Show status every 60 seconds (reduced frequency)

  // Keep the process alive
  process.stdin.resume();
}

function showHelp() {
  console.log(`
🎤 OpenAI Realtime Voice Agent (Terminal Version)

Usage: bun run index.ts [options]
   or: ./yui.sh [options]                # Clean output (filters audio warnings)
   or: bun run start:clean [options]     # Alternative clean output

Options:
  --voice <voice>           Voice to use: alloy, ash, ballad, coral, echo, sage, shimmer, verse (default: alloy)
  --system-prompt <prompt>  System prompt for the AI (default: helpful assistant named Yui)
  --no-audio               Disable audio input/output (connection test mode)
  --help                    Show this help message

Environment Variables:
  OPENAI_API_KEY           Your OpenAI API key (required)

Features:
  - Real-time bidirectional voice conversation via WebSocket
  - Server-side Voice Activity Detection (VAD)
  - Full audio I/O with microphone and speaker support
  - Raw PCM16 audio processing (24kHz, mono)
  - Natural conversation flow with interruption support

Audio Requirements:
  - Microphone access for voice input
  - Speaker/headphones for voice output
  - Audio format: PCM16, 24kHz sample rate, mono channel
  - SoX audio tools (for microphone input)

Clean Output Mode:
  Use ./yui.sh or 'bun run start:clean' to filter out low-level audio warnings
  from the mpg123 library while preserving all important output.

Examples:
  bun run index.ts                                    # Full voice conversation
  ./yui.sh                                            # Clean output mode
  ./yui.sh --voice shimmer                            # Clean output with Shimmer voice
  bun run start:clean --voice coral                   # Alternative clean output
  bun run index.ts --no-audio                         # Test connection only
  bun run index.ts --system-prompt "You are a coding assistant"

Valid Voices:
  alloy, ash, ballad, coral, echo, sage, shimmer, verse

Based on OpenAI Agents JS SDK: https://github.com/openai/openai-agents-js

Note: This version includes full audio I/O support for terminal usage.
Requires microphone and speaker access.
`);
}

// Run the main function
main().catch(console.error);
