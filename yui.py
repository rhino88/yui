#!/usr/bin/env python3

import asyncio
import os
import queue
import sys
import threading
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
try:
	from agents import function_tool  # type: ignore
	from agents.realtime import (
		RealtimeAgent as RTAgent,
		RealtimePlaybackTracker,
		RealtimeRunner,
		RealtimeSession,
		RealtimeSessionEvent,
	)  # type: ignore
except Exception:
	RTAgent = None  # type: ignore
	RealtimeRunner = None  # type: ignore
	RealtimePlaybackTracker = None  # type: ignore
	RealtimeSession = None  # type: ignore
	RealtimeSessionEvent = None  # type: ignore
	def function_tool(fn):  # type: ignore
		return fn
try:
	import sounddevice as sd  # type: ignore
except Exception:
	sd = None  # type: ignore

load_dotenv()


def require_env(name: str) -> str:
	value = os.environ.get(name)
	if not value:
		print(f"❌ {name} environment variable is required", file=sys.stderr)
		sys.exit(1)
	return value


def _truncate_str(s: str, max_length: int) -> str:
	return s if len(s) <= max_length else s[:max_length] + "..."


# Audio configuration
CHUNK_LENGTH_S = 0.04  # 40ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1
ENERGY_THRESHOLD = 0.015
PREBUFFER_CHUNKS = 3
FADE_OUT_MS = 12
# Barge-in robustness
BARGE_IN_FRAMES_REQUIRED = 4  # consecutive frames above threshold while assistant speaks
ECHO_SUPPRESS_MULTIPLIER = 1.8  # mic must exceed playback RMS by this factor


def _sdk_required() -> None:
	if RTAgent is None or RealtimeRunner is None:
		print("❌ Realtime SDK not available. Install with: pip install openai-agents", file=sys.stderr)
		sys.exit(1)
	if sd is None:
		print("❌ sounddevice is required for voice mode. Install with: pip install sounddevice", file=sys.stderr)
		sys.exit(1)


@function_tool
def get_weather(city: str) -> str:
	"""Get the weather in a city."""
	return f"The weather in {city} is sunny."


def create_agent(instructions: str) -> Any:
	return RTAgent(
		name="Yui",
		instructions=instructions,
		tools=[get_weather],
	)


class YuiRealtime:
	def __init__(self, agent: Any, barge_in_mode: str = "rms") -> None:
		self.session: RealtimeSession | None = None
		self.audio_stream: sd.InputStream | None = None
		self.audio_player: sd.OutputStream | None = None
		self.recording = False

		# Playback tracker (optional on older SDKs)
		self.playback_tracker = RealtimePlaybackTracker() if RealtimePlaybackTracker is not None else None

		# Output queue: (samples_np, item_id, content_index)
		self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=0)
		self.interrupt_event = threading.Event()
		self.current_audio_chunk: tuple[np.ndarray[Any, np.dtype[Any]], str, int] | None = None
		self.chunk_position = 0
		self.bytes_per_sample = np.dtype(FORMAT).itemsize

		# Jitter buffer and fade-out state
		self.prebuffering = True
		self.prebuffer_target_chunks = PREBUFFER_CHUNKS
		self.fading = False
		self.fade_total_samples = 0
		self.fade_done_samples = 0
		self.fade_samples = int(SAMPLE_RATE * (FADE_OUT_MS / 1000.0))

		self.agent = agent
		self.barge_in_mode = barge_in_mode  # "server" or "rms"

		# Echo suppression and VAD state
		self.recent_playback_rms: float = 0.0
		self._playback_rms_alpha: float = 0.2  # EMA smoothing for playback RMS
		self.vad_active_frames: int = 0

	def _output_callback(self, outdata, frames: int, time, status) -> None:
		if status:
			print(f"Output callback status: {status}")

		# Interruption with fade-out
		if self.interrupt_event.is_set():
			outdata.fill(0)
			if self.current_audio_chunk is None:
				while not self.output_queue.empty():
					try:
						self.output_queue.get_nowait()
					except queue.Empty:
						break
				self.prebuffering = True
				self.interrupt_event.clear()
				return

			if not self.fading:
				self.fading = True
				self.fade_done_samples = 0
				remaining_in_chunk = len(self.current_audio_chunk[0]) - self.chunk_position
				self.fade_total_samples = min(self.fade_samples, max(0, remaining_in_chunk))

			samples, item_id, content_index = self.current_audio_chunk
			samples_filled = 0
			while (
				samples_filled < len(outdata) and self.fade_done_samples < self.fade_total_samples
			):
				remaining_output = len(outdata) - samples_filled
				remaining_fade = self.fade_total_samples - self.fade_done_samples
				n = min(remaining_output, remaining_fade)

				src = samples[self.chunk_position : self.chunk_position + n].astype(np.float32)
				idx = np.arange(
					self.fade_done_samples, self.fade_done_samples + n, dtype=np.float32
				)
				gain = 1.0 - (idx / float(self.fade_total_samples))
				ramped = np.clip(src * gain, -32768.0, 32767.0).astype(np.int16)
				outdata[samples_filled : samples_filled + n, 0] = ramped

				# Update recent playback RMS for echo suppression
				try:
					play_rms = float(np.sqrt(np.mean((ramped.astype(np.float32) / 32768.0) ** 2)))
					self.recent_playback_rms = (
						(1.0 - self._playback_rms_alpha) * self.recent_playback_rms
						+ self._playback_rms_alpha * play_rms
					)
				except Exception:
					pass

				if self.playback_tracker is not None:
					try:
						self.playback_tracker.on_play_bytes(
							item_id=item_id, item_content_index=content_index, bytes=ramped.tobytes()
						)
					except Exception:
						pass

				samples_filled += n
				self.chunk_position += n
				self.fade_done_samples += n

			if self.fade_done_samples >= self.fade_total_samples:
				self.current_audio_chunk = None
				self.chunk_position = 0
				while not self.output_queue.empty():
					try:
						self.output_queue.get_nowait()
					except queue.Empty:
						break
				self.fading = False
				self.prebuffering = True
				self.interrupt_event.clear()
			return

		# Normal playback path
		outdata.fill(0)
		samples_filled = 0
		while samples_filled < len(outdata):
			if self.current_audio_chunk is None:
				try:
					if (
						self.prebuffering
						and self.output_queue.qsize() < self.prebuffer_target_chunks
					):
						break
					self.prebuffering = False
					self.current_audio_chunk = self.output_queue.get_nowait()
					self.chunk_position = 0
				except queue.Empty:
					break

			remaining_output = len(outdata) - samples_filled
			samples, item_id, content_index = self.current_audio_chunk
			remaining_chunk = len(samples) - self.chunk_position
			samples_to_copy = min(remaining_output, remaining_chunk)

			if samples_to_copy > 0:
				chunk_data = samples[self.chunk_position : self.chunk_position + samples_to_copy]
				outdata[samples_filled : samples_filled + samples_to_copy, 0] = chunk_data
				samples_filled += samples_to_copy
				self.chunk_position += samples_to_copy

				# Update recent playback RMS for echo suppression
				try:
					play_rms = float(np.sqrt(np.mean((chunk_data.astype(np.float32) / 32768.0) ** 2)))
					self.recent_playback_rms = (
						(1.0 - self._playback_rms_alpha) * self.recent_playback_rms
						+ self._playback_rms_alpha * play_rms
					)
				except Exception:
					pass

				if self.playback_tracker is not None:
					try:
						self.playback_tracker.on_play_bytes(
							item_id=item_id,
							item_content_index=content_index,
							bytes=chunk_data.tobytes(),
						)
					except Exception:
						pass

				if self.chunk_position >= len(samples):
					self.current_audio_chunk = None
					self.chunk_position = 0

	async def run(self, voice_name: str) -> None:
		print("Connecting, may take a few seconds...")

		chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
		self.audio_player = sd.OutputStream(
			channels=CHANNELS,
			samplerate=SAMPLE_RATE,
			dtype=FORMAT,
			callback=self._output_callback,
			blocksize=chunk_size,
		)
		self.audio_player.start()

		try:
			runner = RealtimeRunner(self.agent)
			model_config: Any = {
				"initial_model_settings": {
					"turn_detection": {
						"type": "semantic_vad",
						"interrupt_response": True,
						"create_response": True,
					},
					"voice": voice_name,
				},
			}
			# Only include playback tracker if available
			if self.playback_tracker is not None:
				model_config["playback_tracker"] = self.playback_tracker
			async with await runner.run(model_config=model_config) as session:
				self.session = session
				print("Connected. Starting audio recording...")
				await self.start_audio_recording()
				print("Audio recording started. Speak when ready.")

				async for event in session:
					await self._on_event(event)
		finally:
			if self.audio_player and self.audio_player.active:
				self.audio_player.stop()
			if self.audio_player:
				self.audio_player.close()
		print("Session ended")

	async def start_audio_recording(self) -> None:
		self.audio_stream = sd.InputStream(
			channels=CHANNELS,
			samplerate=SAMPLE_RATE,
			dtype=FORMAT,
		)
		self.audio_stream.start()
		self.recording = True
		asyncio.create_task(self.capture_audio())

	async def capture_audio(self) -> None:
		if not self.audio_stream or not self.session:
			return

		read_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)

		try:
			def rms_energy(samples: np.ndarray[Any, np.dtype[Any]]) -> float:
				if samples.size == 0:
					return 0.0
				x = samples.astype(np.float32) / 32768.0
				return float(np.sqrt(np.mean(x * x)))

			while self.recording:
				if self.audio_stream.read_available < read_size:
					await asyncio.sleep(0.01)
					continue

				data, _ = self.audio_stream.read(read_size)
				audio_bytes = data.tobytes()

				assistant_playing = (
					self.current_audio_chunk is not None or not self.output_queue.empty()
				)
				samples = data.reshape(-1)
				mic_rms = rms_energy(samples)
				if self.barge_in_mode == "server":
					# Always stream mic to server; rely on server-side semantic VAD + interruption
					self.vad_active_frames = 0
					await self.session.send_audio(audio_bytes)
				else:
					# Local RMS-based barge-in (with echo suppression)
					if assistant_playing:
						playback_guard = max(ENERGY_THRESHOLD, self.recent_playback_rms * ECHO_SUPPRESS_MULTIPLIER)
						if mic_rms >= playback_guard:
							self.vad_active_frames += 1
							if self.vad_active_frames >= BARGE_IN_FRAMES_REQUIRED:
								self.interrupt_event.set()
								await self.session.send_audio(audio_bytes)
						else:
							self.vad_active_frames = 0
					else:
						self.vad_active_frames = 0
						await self.session.send_audio(audio_bytes)

				await asyncio.sleep(0)
		except Exception as e:
			print(f"Audio capture error: {e}")
		finally:
			if self.audio_stream and self.audio_stream.active:
				self.audio_stream.stop()
			if self.audio_stream:
				self.audio_stream.close()

	async def _on_event(self, event: RealtimeSessionEvent) -> None:
		try:
			if event.type == "agent_start":
				print(f"Agent started: {event.agent.name}")
			elif event.type == "agent_end":
				print(f"Agent ended: {event.agent.name}")
			elif event.type == "handoff":
				print(f"Handoff from {event.from_agent.name} to {event.to_agent.name}")
			elif event.type == "tool_start":
				print(f"Tool started: {event.tool.name}")
			elif event.type == "tool_end":
				print(f"Tool ended: {event.tool.name}; output: {event.output}")
			elif event.type == "audio_end":
				print("Audio ended")
			elif event.type == "audio":
				np_audio = np.frombuffer(event.audio.data, dtype=np.int16)
				self.output_queue.put_nowait((np_audio, event.item_id, event.content_index))
			elif event.type == "audio_interrupted":
				print("Audio interrupted")
				self.prebuffering = True
				self.interrupt_event.set()
			elif event.type == "error":
				print(f"Error: {event.error}")
			elif event.type in ("history_updated", "history_added"):
				pass
			elif event.type == "raw_model_event":
				print(f"Raw model event: {_truncate_str(str(event.data), 200)}")
			else:
				print(f"Unknown event type: {event.type}")
		except Exception as e:
			print(f"Error processing event: {_truncate_str(str(e), 200)}")


async def main():
	# Parse args: --system-prompt "..." --voice <name>
	args = sys.argv[1:]
	system_prompt: Optional[str] = None
	voice_name = "marin"
	i = 0
	while i < len(args):
		if args[i] == "--system-prompt" and i + 1 < len(args):
			system_prompt = args[i + 1]
			i += 2
		elif args[i] == "--voice" and i + 1 < len(args):
			voice_name = args[i + 1]
			i += 2
		elif args[i] in ("--help", "-h"):
			print("""
🎤 Yui (Python) - Realtime Agent

Usage: python yui.py [--system-prompt PROMPT] [--voice VOICE_NAME]

Environment:
  OPENAI_API_KEY          Required

Behavior:
  - Runs OpenAI Agents SDK Realtime agent and streams audio responses
			""")
			return
		else:
			print(f"❌ Unknown argument: {args[i]}")
			return

	require_env("OPENAI_API_KEY")
	_sdk_required()
	instructions = system_prompt or "You always greet the user with 'Top of the morning to you'."
	agent = create_agent(instructions)
	demo = YuiRealtime(agent, barge_in_mode="rms")
	await demo.run(voice_name)


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		print("\nExiting...")
		sys.exit(0)
