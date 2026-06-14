#!/usr/bin/env python3

import asyncio
import base64
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from dotenv import load_dotenv

try:
	import websockets  # type: ignore
	from websockets.asyncio.client import ClientConnection  # type: ignore
except Exception:
	websockets = None  # type: ignore
	ClientConnection = Any  # type: ignore

try:
	import sounddevice as sd  # type: ignore
except Exception:
	sd = None  # type: ignore

try:
	from scipy.signal import resample_poly  # type: ignore
except Exception:
	resample_poly = None  # type: ignore

try:
	from openwakeword.model import Model as OWWModel  # type: ignore
except Exception:
	OWWModel = None  # type: ignore

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
# End-of-utterance detection (rms mode) and server VAD config
DEFAULT_SILENCE_MS = 750  # silence after speech that ends a user turn
MIN_UTTERANCE_MS = 300  # minimum speech length before we'll commit a turn

# xAI Voice Agent API
XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL = "grok-voice-think-fast-1.0"
DEFAULT_VOICE = "Ara"
INPUT_TRANSCRIPTION_MODEL = "grok-2-audio"
KNOWN_VOICES = {"Eve", "Ara", "Leo", "Rex", "Sal"}

# Wake word
WAKE_WORD_SAMPLE_RATE = 16000  # openwakeword requirement
DEFAULT_WAKE_WORD_MODEL = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "hey_yoo_wee.onnx"
)
DEFAULT_WAKE_WORD_THRESHOLD = 0.5
DEFAULT_SLEEP_AFTER_S = 30.0
# Whole-word phrases that, when they appear anywhere in an utterance, tell Yui to
# go back to sleep. Matching is on word boundaries (see _is_sleep_command), so
# "stop" matches "stop talking yui" but not "nonstop", and these stay liberal on
# purpose — a false sleep just means saying the wake word again.
SLEEP_COMMAND_PHRASES = {
	"goodbye",
	"good bye",
	"good night",
	"goodnight",
	"night night",
	"bye",
	"bye bye",
	"stop",
	"go to sleep",
	"go back to sleep",
	"back to sleep",
	"go to bed",
	"go away",
	"shut up",
	"shush",
	"be quiet",
	"quiet",
	"leave me alone",
	"nevermind",
	"never mind",
	"that's all",
	"that is all",
	"that'll be all",
	"that will be all",
	"i'm done",
	"i am done",
	"we're done",
	"we are done",
}
# Note: response.create.response.instructions OVERRIDES session instructions for
# the turn — it does NOT merge. So the greeting prompt below has to carry any
# onboarding/memory-aware directives inline; see _compose_greeting_instructions.

# Memory
DEFAULT_MEMORY_PATH = str(Path.home() / ".yui" / "memory.json")


def _utcnow_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_local() -> datetime:
	return datetime.now().astimezone()


def _now_human() -> str:
	# e.g. "Friday, April 25, 2026, 12:34 PM PDT"
	return _now_local().strftime("%A, %B %d, %Y, %I:%M %p %Z").strip()


def _tool_current_datetime() -> dict[str, Any]:
	now = _now_local()
	return {
		"iso": now.isoformat(timespec="seconds"),
		"human": _now_human(),
		"date": now.date().isoformat(),
		"weekday": now.strftime("%A"),
		"timezone": str(now.tzinfo) if now.tzinfo else "local",
	}


class MemoryStore:
	"""Persistent per-user memory backed by a single JSON file."""

	def __init__(self, path: str) -> None:
		self.path = Path(path).expanduser()
		self.data: dict[str, Any] = {
			"profile": {},
			"observations": [],
			"onboarded": False,
			"created_at": _utcnow_iso(),
			"updated_at": _utcnow_iso(),
		}
		self._load()

	def _load(self) -> None:
		if not self.path.exists():
			return
		try:
			with self.path.open("r", encoding="utf-8") as f:
				loaded = json.load(f)
			if isinstance(loaded, dict):
				# Merge over defaults so older files pick up new keys.
				self.data.update(loaded)
				self.data.setdefault("profile", {})
				self.data.setdefault("observations", [])
				self.data.setdefault("onboarded", False)
		except Exception as e:
			print(f"⚠️  Failed to load memory at {self.path}: {e}", file=sys.stderr)

	def _save(self) -> None:
		self.data["updated_at"] = _utcnow_iso()
		try:
			self.path.parent.mkdir(parents=True, exist_ok=True)
			# Atomic write: tmp file in same dir, then rename.
			fd, tmp_path = tempfile.mkstemp(prefix=".memory.", suffix=".json", dir=self.path.parent)
			try:
				with os.fdopen(fd, "w", encoding="utf-8") as f:
					json.dump(self.data, f, ensure_ascii=False, indent=2)
				os.replace(tmp_path, self.path)
			except Exception:
				try:
					os.unlink(tmp_path)
				except Exception:
					pass
				raise
		except Exception as e:
			print(f"⚠️  Failed to save memory: {e}", file=sys.stderr)

	# ----- Mutators -----

	def set_profile_field(self, field: str, value: str) -> dict[str, Any]:
		field = (field or "").strip().lower().replace(" ", "_")
		if not field:
			return {"ok": False, "error": "field is required"}
		self.data["profile"][field] = value
		self._save()
		return {"ok": True, "field": field, "value": value}

	def add_observation(self, text: str) -> dict[str, Any]:
		text = (text or "").strip()
		if not text:
			return {"ok": False, "error": "text is required"}
		self.data["observations"].append({"text": text, "captured_at": _utcnow_iso()})
		self._save()
		return {"ok": True, "count": len(self.data["observations"])}

	def mark_onboarded(self) -> dict[str, Any]:
		self.data["onboarded"] = True
		self._save()
		return {"ok": True}

	# ----- Read-only -----

	def is_onboarded(self) -> bool:
		return bool(self.data.get("onboarded"))

	def as_prompt_context(self) -> str:
		profile = self.data.get("profile") or {}
		observations = self.data.get("observations") or []
		lines: list[str] = []
		lines.append("=== Memory about the user ===")
		lines.append(f"onboarded: {str(bool(self.data.get('onboarded'))).lower()}")
		if profile:
			lines.append("Profile:")
			for k, v in profile.items():
				lines.append(f"  - {k}: {v}")
		else:
			lines.append("Profile: (empty)")
		if observations:
			lines.append("Observations (most recent last):")
			# Cap context size — keep the last ~20 to avoid bloating the prompt.
			for obs in observations[-20:]:
				text = obs.get("text") if isinstance(obs, dict) else str(obs)
				lines.append(f"  - {text}")
		else:
			lines.append("Observations: (none yet)")
		lines.append("=== End memory ===")
		return "\n".join(lines)


def _normalize_voice(name: str) -> str:
	# Server expects capitalized voice IDs; accept any casing from the CLI.
	cap = name.strip().capitalize()
	return cap if cap in KNOWN_VOICES else name


def _deps_required() -> None:
	if websockets is None:
		print("❌ websockets is required. Install with: pip install websockets", file=sys.stderr)
		sys.exit(1)
	if sd is None:
		print("❌ sounddevice is required. Install with: pip install sounddevice", file=sys.stderr)
		sys.exit(1)


# ---------- Google APIs ----------
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_WEATHER_URL = "https://weather.googleapis.com/v1/currentConditions:lookup"
GOOGLE_POLLEN_URL = "https://pollen.googleapis.com/v1/forecast:lookup"
GOOGLE_AIR_QUALITY_URL = "https://airquality.googleapis.com/v1/currentConditions:lookup"
HTTP_TIMEOUT_S = 8.0

_geocode_cache: dict[str, tuple[float, float, str]] = {}


def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
	full = f"{url}?{urllib.parse.urlencode(params)}"
	req = urllib.request.Request(full, headers={"Accept": "application/json"})
	with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _http_post_json(
	url: str, params: dict[str, Any], body: dict[str, Any]
) -> dict[str, Any]:
	full = f"{url}?{urllib.parse.urlencode(params)}"
	data = json.dumps(body).encode("utf-8")
	req = urllib.request.Request(
		full,
		data=data,
		headers={"Accept": "application/json", "Content-Type": "application/json"},
	)
	with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _http_error_message(e: BaseException) -> str:
	if isinstance(e, urllib.error.HTTPError):
		try:
			body = e.read().decode("utf-8", errors="replace")
		except Exception:
			body = ""
		return f"HTTP {e.code}: {_truncate_str(body, 240)}"
	return str(e)


def _geocode_sync(location: str, key: str) -> Optional[tuple[float, float, str]]:
	cache_key = location.strip().lower()
	if cache_key in _geocode_cache:
		return _geocode_cache[cache_key]
	try:
		data = _http_get_json(GOOGLE_GEOCODE_URL, {"address": location, "key": key})
	except Exception:
		return None
	results = data.get("results") or []
	if not results:
		return None
	loc = ((results[0].get("geometry") or {}).get("location")) or {}
	lat = loc.get("lat")
	lng = loc.get("lng")
	formatted = results[0].get("formatted_address") or location
	if lat is None or lng is None:
		return None
	pair = (float(lat), float(lng), str(formatted))
	_geocode_cache[cache_key] = pair
	return pair


async def _resolve_location(location: str) -> Optional[tuple[float, float, str]]:
	key = os.environ.get("GOOGLE_API_KEY")
	if not key:
		return None
	return await asyncio.to_thread(_geocode_sync, location, key)


def _require_google_key() -> Optional[dict[str, Any]]:
	if not os.environ.get("GOOGLE_API_KEY"):
		return {"error": "GOOGLE_API_KEY not set in environment"}
	return None


async def _tool_weather(location: str) -> dict[str, Any]:
	err = _require_google_key()
	if err:
		return err
	key = os.environ["GOOGLE_API_KEY"]
	coords = await _resolve_location(location)
	if coords is None:
		return {"error": f"could not geocode '{location}'"}
	lat, lng, formatted = coords
	try:
		data = await asyncio.to_thread(
			_http_get_json,
			GOOGLE_WEATHER_URL,
			{"key": key, "location.latitude": lat, "location.longitude": lng},
		)
	except Exception as e:
		return {"error": _http_error_message(e)}
	temp_block = data.get("temperature") or {}
	feels_block = data.get("feelsLikeTemperature") or {}
	condition = (data.get("weatherCondition") or {})
	return {
		"location": formatted,
		"temperature": temp_block.get("degrees"),
		"feels_like": feels_block.get("degrees"),
		"unit": temp_block.get("unit"),
		"humidity_percent": data.get("relativeHumidity"),
		"conditions": ((condition.get("description") or {}).get("text")),
		"wind_kph": ((data.get("wind") or {}).get("speed") or {}).get("value"),
	}


async def _tool_pollen(location: str) -> dict[str, Any]:
	err = _require_google_key()
	if err:
		return err
	key = os.environ["GOOGLE_API_KEY"]
	coords = await _resolve_location(location)
	if coords is None:
		return {"error": f"could not geocode '{location}'"}
	lat, lng, formatted = coords
	try:
		data = await asyncio.to_thread(
			_http_get_json,
			GOOGLE_POLLEN_URL,
			{
				"key": key,
				"location.latitude": lat,
				"location.longitude": lng,
				"days": 1,
			},
		)
	except Exception as e:
		return {"error": _http_error_message(e)}
	daily = (data.get("dailyInfo") or [{}])[0]
	pollen_types = daily.get("pollenTypeInfo") or []
	summary = []
	for pt in pollen_types:
		idx = pt.get("indexInfo") or {}
		summary.append(
			{
				"type": pt.get("displayName") or pt.get("code"),
				"category": idx.get("category"),
				"value": idx.get("value"),
			}
		)
	return {"location": formatted, "pollen": summary}


async def _tool_air_quality(location: str) -> dict[str, Any]:
	err = _require_google_key()
	if err:
		return err
	key = os.environ["GOOGLE_API_KEY"]
	coords = await _resolve_location(location)
	if coords is None:
		return {"error": f"could not geocode '{location}'"}
	lat, lng, formatted = coords
	try:
		data = await asyncio.to_thread(
			_http_post_json,
			GOOGLE_AIR_QUALITY_URL,
			{"key": key},
			{"location": {"latitude": lat, "longitude": lng}},
		)
	except Exception as e:
		return {"error": _http_error_message(e)}
	indexes = data.get("indexes") or []
	primary = indexes[0] if indexes else {}
	return {
		"location": formatted,
		"aqi": primary.get("aqi"),
		"category": primary.get("category"),
		"dominant_pollutant": primary.get("dominantPollutant"),
		"index_name": primary.get("displayName"),
	}


# Tool registry built per-session, since tools close over a MemoryStore.


def _build_tools(
	memory: "MemoryStore",
) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
	return {
		"get_current_datetime": (
			{
				"type": "function",
				"name": "get_current_datetime",
				"description": (
					"Get the current local date, time, weekday, and timezone. "
					"Use this when the user asks 'what day is it', 'what time is it', "
					"or for anything that depends on the current moment (e.g. age "
					"calculations, scheduling)."
				),
				"parameters": {"type": "object", "properties": {}},
			},
			_tool_current_datetime,
		),
		"get_weather": (
			{
				"type": "function",
				"name": "get_weather",
				"description": (
					"Get current weather (temperature, conditions, humidity, wind) "
					"for a city, address, or place name. Pass a free-form location "
					"string; geocoding is handled automatically."
				),
				"parameters": {
					"type": "object",
					"properties": {
						"location": {
							"type": "string",
							"description": "City, address, or place name (e.g. 'San Francisco' or 'Times Square, NYC').",
						}
					},
					"required": ["location"],
				},
			},
			_tool_weather,
		),
		"get_pollen": (
			{
				"type": "function",
				"name": "get_pollen",
				"description": (
					"Get today's pollen forecast (grass, tree, weed levels) for a "
					"location. Useful when the user mentions allergies."
				),
				"parameters": {
					"type": "object",
					"properties": {
						"location": {
							"type": "string",
							"description": "City, address, or place name.",
						}
					},
					"required": ["location"],
				},
			},
			_tool_pollen,
		),
		"get_air_quality": (
			{
				"type": "function",
				"name": "get_air_quality",
				"description": (
					"Get current air quality index (AQI) and dominant pollutant "
					"for a location."
				),
				"parameters": {
					"type": "object",
					"properties": {
						"location": {
							"type": "string",
							"description": "City, address, or place name.",
						}
					},
					"required": ["location"],
				},
			},
			_tool_air_quality,
		),
		"set_profile_field": (
			{
				"type": "function",
				"name": "set_profile_field",
				"description": (
					"Save a structured fact about the user (name, birthday, pronouns, "
					"timezone, location, etc.) to long-term memory. Use lowercase "
					"snake_case for the field name. Birthdays should be ISO format "
					"(YYYY-MM-DD) when known, otherwise free-form."
				),
				"parameters": {
					"type": "object",
					"properties": {
						"field": {
							"type": "string",
							"description": "Field name, lowercase snake_case (e.g. 'name', 'birthday', 'timezone').",
						},
						"value": {
							"type": "string",
							"description": "The value to store.",
						},
					},
					"required": ["field", "value"],
				},
			},
			memory.set_profile_field,
		),
		"add_observation": (
			{
				"type": "function",
				"name": "add_observation",
				"description": (
					"Append a free-form observation about the user to long-term "
					"memory (preferences, interests, relationships, recent events, "
					"things they care about). Use this naturally as you learn "
					"about them — do not announce that you are doing it."
				),
				"parameters": {
					"type": "object",
					"properties": {
						"text": {
							"type": "string",
							"description": "One short sentence describing what you learned.",
						},
					},
					"required": ["text"],
				},
			},
			memory.add_observation,
		),
		"mark_onboarded": (
			{
				"type": "function",
				"name": "mark_onboarded",
				"description": (
					"Call this once you have collected the user's name and birthday "
					"during the first-time onboarding. Do not call it before those "
					"are saved via set_profile_field."
				),
				"parameters": {"type": "object", "properties": {}},
			},
			lambda: memory.mark_onboarded(),
		),
	}


class YuiRealtime:
	def __init__(
		self,
		instructions: str,
		memory: MemoryStore,
		voice: str = DEFAULT_VOICE,
		model: str = DEFAULT_MODEL,
		barge_in_mode: str = "rms",
		wake_word_model_path: Optional[str] = None,
		wake_word_threshold: float = DEFAULT_WAKE_WORD_THRESHOLD,
		sleep_after_s: float = DEFAULT_SLEEP_AFTER_S,
		greet_on_wake: bool = True,
		silence_ms: int = DEFAULT_SILENCE_MS,
	) -> None:
		self.memory = memory

		# Wake-word state (set before instructions/tools — both depend on it).
		self.wake_word_model_path = wake_word_model_path
		self.wake_word_threshold = wake_word_threshold
		self.sleep_after_s = sleep_after_s
		self.greet_on_wake = greet_on_wake
		self.wake_model: Any = None  # populated in run()
		self.awake: bool = wake_word_model_path is None
		self.last_speech_at: float = time.monotonic()

		self.tools = _build_tools(memory)
		# When wake-word gating is on, give the model a tool to end the
		# conversation itself — a backstop for sleep phrasings the local
		# keyword matcher (_is_sleep_command) doesn't catch.
		if self.wake_word_model_path is not None:
			self.tools["go_to_sleep"] = (
				{
					"type": "function",
					"name": "go_to_sleep",
					"description": (
						"End the conversation and go back to sleep until the user "
						"says the wake word again. Call this the moment the user "
						"signals they are done or want you to stop — e.g. 'goodbye', "
						"'good night', 'stop talking', \"that's all\", 'go to sleep', "
						"'leave me alone'. Do not call it mid-task or just because the "
						"user paused; only on a clear signal they want you to stop."
					),
					"parameters": {"type": "object", "properties": {}},
				},
				self._tool_go_to_sleep,
			)
		self.base_instructions = instructions
		self.instructions = self._compose_instructions()
		self.voice = voice
		self.model = model
		self.barge_in_mode = barge_in_mode  # "server" or "rms"
		self.silence_ms = max(200, int(silence_ms))

		self.ws: Optional[ClientConnection] = None
		self.audio_stream: Optional["sd.InputStream"] = None
		self.audio_player: Optional["sd.OutputStream"] = None
		self.recording = False

		# Output queue: (samples_np, item_id, content_index)
		self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=0)
		self.interrupt_event = threading.Event()
		self.current_audio_chunk: Optional[tuple[np.ndarray, str, int]] = None
		self.chunk_position = 0
		self.bytes_per_sample = np.dtype(FORMAT).itemsize

		# Jitter buffer and fade-out state
		self.prebuffering = True
		self.prebuffer_target_chunks = PREBUFFER_CHUNKS
		self.fading = False
		self.fade_total_samples = 0
		self.fade_done_samples = 0
		self.fade_samples = int(SAMPLE_RATE * (FADE_OUT_MS / 1000.0))

		# Echo suppression / VAD state
		self.recent_playback_rms: float = 0.0
		self._playback_rms_alpha: float = 0.2
		self.vad_active_frames: int = 0
		self.force_barge_in: bool = False

		# Track in-flight assistant response so we can cancel on barge-in
		self.active_response_id: Optional[str] = None
		self.active_item_id: str = ""
		self.active_content_index: int = 0

		# Async send lock to serialize WebSocket sends
		self._send_lock = asyncio.Lock()

	# ---------- Audio output callback ----------

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

				try:
					play_rms = float(np.sqrt(np.mean((ramped.astype(np.float32) / 32768.0) ** 2)))
					self.recent_playback_rms = (
						(1.0 - self._playback_rms_alpha) * self.recent_playback_rms
						+ self._playback_rms_alpha * play_rms
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

				try:
					play_rms = float(np.sqrt(np.mean((chunk_data.astype(np.float32) / 32768.0) ** 2)))
					self.recent_playback_rms = (
						(1.0 - self._playback_rms_alpha) * self.recent_playback_rms
						+ self._playback_rms_alpha * play_rms
					)
				except Exception:
					pass

				if self.chunk_position >= len(samples):
					self.current_audio_chunk = None
					self.chunk_position = 0

	# ---------- WebSocket send helpers ----------

	async def _send(self, event: dict[str, Any]) -> None:
		if self.ws is None:
			return
		async with self._send_lock:
			await self.ws.send(json.dumps(event))

	def _compose_instructions(self) -> str:
		parts = [
			self.base_instructions,
			"",
			f"Current local date/time at session start: {_now_human()}. "
			"Call `get_current_datetime` for an updated value if needed.",
			"",
			self.memory.as_prompt_context(),
			"",
		]
		if not self.memory.is_onboarded():
			parts.append(
				"FIRST-TIME ONBOARDING: This is the user's first conversation with you. "
				"Warmly introduce yourself and ask for their name and birthday "
				"(birthday for fun things like birthday wishes; tell them they can "
				"skip if they prefer). For each fact you learn, call the "
				"`set_profile_field` tool (e.g. field='name', field='birthday'). "
				"Once you have at minimum their name, call `mark_onboarded`. "
				"Do not pepper them with questions — keep it conversational."
			)
		else:
			parts.append(
				"The user is already onboarded. Do NOT ask for their name or "
				"birthday again. Use what is already in the memory section above."
			)
		parts.append("")
		parts.append(
			"ONGOING MEMORY: As you learn personal facts about the user during "
			"conversation (preferences, interests, relationships, recent events, "
			"context that would help future conversations), call the "
			"`add_observation` tool with one short sentence. For structured facts "
			"(name, location, timezone, pronouns, job, etc.) prefer "
			"`set_profile_field`. Do this naturally and silently — never announce "
			"that you are saving anything. Do not store secrets, payment info, or "
			"anything the user explicitly asks you not to remember."
		)
		parts.append(
			"WEB SEARCH: You have access to the built-in `web_search` tool. Use it "
			"when the user asks about current events, recent facts, live web "
			"information, or anything likely to have changed since your training data."
		)
		if self.wake_word_model_path is not None:
			parts.append(
				"ENDING THE CONVERSATION: When the user clearly signals they are "
				"done or want you to stop — saying things like 'goodbye', 'good "
				"night', 'stop talking', \"that's all\", 'go to sleep', or 'leave me "
				"alone' — call the `go_to_sleep` tool to end the conversation. You "
				"may give a brief sign-off first if it feels natural, but keep it to "
				"a few words. Do not call it mid-task or merely because the user went "
				"quiet."
			)
		return "\n".join(parts)

	async def _send_session_update(self) -> None:
		# In "rms" mode we drive turn-taking locally and disable server VAD so the
		# server doesn't auto-start responses while we gate the mic.
		turn_detection: Any
		if self.barge_in_mode == "server":
			turn_detection = {
				"type": "server_vad",
				"silence_duration_ms": self.silence_ms,
				"prefix_padding_ms": 300,
			}
		else:
			turn_detection = None

		session: dict[str, Any] = {
			"voice": self.voice,
			"instructions": self.instructions,
			"turn_detection": turn_detection,
			"input_audio_transcription": {"model": INPUT_TRANSCRIPTION_MODEL},
			"audio": {
				"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
				"output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
			},
			"tools": [
				{"type": "web_search"},
				*[schema for schema, _ in self.tools.values()],
			],
		}
		await self._send({"type": "session.update", "session": session})

	async def _send_audio_chunk(self, audio_bytes: bytes) -> None:
		await self._send(
			{
				"type": "input_audio_buffer.append",
				"audio": base64.b64encode(audio_bytes).decode("ascii"),
			}
		)

	async def _commit_and_respond(self) -> None:
		# Manual turn boundary used by RMS-mode barge-in/end-of-utterance triggers.
		await self._send({"type": "input_audio_buffer.commit"})
		await self._send({"type": "response.create"})

	async def _cancel_response(self) -> None:
		if self.active_response_id is None:
			return
		await self._send({"type": "response.cancel"})
		self.active_response_id = None

	# ---------- Wake / sleep transitions ----------

	def _load_wake_model(self) -> None:
		if self.wake_word_model_path is None:
			return
		if OWWModel is None:
			print(
				"⚠️  openwakeword not installed; running without wake word. Install with: pip install openwakeword",
				file=sys.stderr,
			)
			self.wake_word_model_path = None
			self.awake = True
			return
		if resample_poly is None:
			print(
				"⚠️  scipy not installed; running without wake word. Install with: pip install scipy",
				file=sys.stderr,
			)
			self.wake_word_model_path = None
			self.awake = True
			return
		if not os.path.exists(self.wake_word_model_path):
			print(
				f"⚠️  Wake-word model not found at {self.wake_word_model_path}; running without wake word.",
				file=sys.stderr,
			)
			self.wake_word_model_path = None
			self.awake = True
			return
		def _instantiate() -> Any:
			return OWWModel(
				wakeword_models=[self.wake_word_model_path],
				inference_framework="onnx",
			)

		try:
			self.wake_model = _instantiate()
		except Exception as e:
			# openwakeword needs bundled preprocessor models (melspectrogram + embedding).
			# They aren't installed by default — fetch them once and retry.
			msg = str(e)
			if "melspectrogram" in msg or "embedding" in msg or "NO_SUCHFILE" in msg:
				print("⏬ Downloading openwakeword preprocessor models (one-time)...", file=sys.stderr)
				try:
					import openwakeword.utils  # type: ignore
					openwakeword.utils.download_models()
					self.wake_model = _instantiate()
				except Exception as e2:
					print(f"⚠️  Failed to download/load wake-word model: {e2}", file=sys.stderr)
					self.wake_word_model_path = None
					self.wake_model = None
					self.awake = True
					return
			else:
				print(f"⚠️  Failed to load wake-word model: {e}", file=sys.stderr)
				self.wake_word_model_path = None
				self.wake_model = None
				self.awake = True
				return
		print(f"💤 Asleep. Say the wake word to activate (model={os.path.basename(self.wake_word_model_path)})")

	def _compose_greeting_instructions(self) -> str:
		profile = self.memory.data.get("profile") or {}
		name = profile.get("name")
		now_line = f"Current local date/time: {_now_human()}.\n\n"
		if not self.memory.is_onboarded():
			return now_line + (
				"The user just woke you with the wake word and this is their FIRST "
				"conversation with you. Do all of the following:\n"
				"1. Greet them warmly in one short sentence and introduce yourself as Yui.\n"
				"2. Ask for their name. After they answer, call the `set_profile_field` "
				"tool with field='name'.\n"
				"3. Then ask for their birthday in a separate turn — tell them they "
				"can skip if they'd rather not share. If they answer, call "
				"`set_profile_field` with field='birthday' (ISO YYYY-MM-DD if known).\n"
				"4. Once you have at least their name saved, call the "
				"`mark_onboarded` tool. Do NOT call it before saving the name.\n"
				"Keep each turn short and conversational. Do not ask multiple "
				"questions in one turn. Do not announce that you're saving things."
			)
		if name:
			return now_line + (
				f"The user just woke you with the wake word. You already know them "
				f"as {name}. Greet {name} warmly and briefly by name in one short "
				"sentence, then wait for their request. Do NOT ask for their name "
				"or birthday again — those are already saved."
			)
		return now_line + (
			"The user just woke you with the wake word. Greet them briefly and "
			"warmly in one short sentence, then wait for their request."
		)

	async def _wake_up(self) -> None:
		if self.awake:
			return
		self.awake = True
		self.last_speech_at = time.monotonic()
		if self.wake_model is not None:
			try:
				self.wake_model.reset()
			except Exception:
				pass
		state = "onboarding" if not self.memory.is_onboarded() else "known user"
		print(f"🟢 Awake ({state})")
		# Drop anything the server may have buffered while we were ignoring it.
		await self._send({"type": "input_audio_buffer.clear"})
		if self.greet_on_wake:
			await self._send(
				{
					"type": "response.create",
					"response": {"instructions": self._compose_greeting_instructions()},
				}
			)

	async def _go_to_sleep(self) -> None:
		if not self.awake or self.wake_word_model_path is None:
			return
		self.awake = False
		print("💤 Sleeping (say the wake word again)")
		# Stop any in-flight assistant audio and clear local + server buffers.
		self.interrupt_event.set()
		await self._cancel_response()
		await self._send({"type": "input_audio_buffer.clear"})
		if self.wake_model is not None:
			try:
				self.wake_model.reset()
			except Exception:
				pass

	async def _tool_go_to_sleep(self) -> dict[str, str]:
		"""Tool handler: the model decided the user wants to end the conversation."""
		await self._go_to_sleep()
		return {"status": "asleep"}

	def _is_sleep_command(self, transcript: str) -> bool:
		"""Return True when the user's utterance is a request to go back to sleep.

		Matches a sleep phrase appearing anywhere in the utterance on word
		boundaries, so loose phrasings like "stop talking yui", "okay yui good
		night", or "alright that's all for now" all trigger sleep.
		"""
		# Keep only letters/apostrophes as words; everything else is a boundary.
		words = re.findall(r"[a-z']+", transcript.lower())
		if not words:
			return False
		# Pad so phrase lookups are whole-word: " stop " won't match "nonstop".
		joined = " " + " ".join(words) + " "
		return any(f" {phrase} " in joined for phrase in SLEEP_COMMAND_PHRASES)

	def _detect_wake_word(self, samples_24k: np.ndarray) -> bool:
		"""Resample 24 kHz → 16 kHz and run one wake-word inference step."""
		if self.wake_model is None or resample_poly is None:
			return False
		try:
			# 24 kHz -> 16 kHz: factor 2/3
			resampled = resample_poly(samples_24k.astype(np.float32), 2, 3)
			pcm16 = np.clip(resampled, -32768.0, 32767.0).astype(np.int16)
			scores = self.wake_model.predict(pcm16)
		except Exception as e:
			print(f"wake-word predict error: {_truncate_str(str(e), 200)}")
			return False
		if not scores:
			return False
		best = max(scores.values())
		return best >= self.wake_word_threshold

	# ---------- Run loop ----------

	async def run(self) -> None:
		print("Connecting to xAI Voice Agent...")

		api_key = require_env("XAI_API_KEY")
		url = f"{XAI_REALTIME_URL}?model={self.model}"

		self._load_wake_model()

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
			async with websockets.connect(
				url,
				additional_headers={"Authorization": f"Bearer {api_key}"},
				max_size=None,
			) as ws:
				self.ws = ws
				await self._send_session_update()
				print(f"Connected. Model={self.model} voice={self.voice}")

				await self.start_audio_recording()
				asyncio.create_task(self._keyboard_interrupt_listener())
				print("Audio recording started. Speak when ready.")

				async for raw in ws:
					try:
						event = json.loads(raw)
					except Exception:
						continue
					await self._on_event(event)
		finally:
			self.recording = False
			if self.audio_player and self.audio_player.active:
				self.audio_player.stop()
			if self.audio_player:
				self.audio_player.close()
			print("Session ended")

	# ---------- Audio capture ----------

	async def start_audio_recording(self) -> None:
		self.audio_stream = sd.InputStream(
			channels=CHANNELS,
			samplerate=SAMPLE_RATE,
			dtype=FORMAT,
		)
		self.audio_stream.start()
		self.recording = True
		asyncio.create_task(self.capture_audio())

	async def _keyboard_interrupt_listener(self) -> None:
		"""Listen for Enter key to trigger manual barge-in."""
		# Disable when stdin isn't a TTY (e.g. piped or /dev/null) — readline
		# returns "" forever and would busy-loop firing barge-in.
		if not sys.stdin.isatty():
			return
		loop = asyncio.get_running_loop()
		while self.ws is not None:
			try:
				line = await loop.run_in_executor(None, sys.stdin.readline)
				if line == "":
					# EOF — stdin closed.
					return
				self.force_barge_in = True
				self.interrupt_event.set()
				await self._cancel_response()
			except Exception:
				await asyncio.sleep(0.1)

	async def capture_audio(self) -> None:
		if not self.audio_stream or not self.ws:
			return

		read_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)

		def rms_energy(samples: np.ndarray) -> float:
			if samples.size == 0:
				return 0.0
			x = samples.astype(np.float32) / 32768.0
			return float(np.sqrt(np.mean(x * x)))

		# Track whether we've sent any audio in the current user turn (rms mode)
		turn_has_audio = False
		silence_chunks = 0
		speech_chunks = 0
		# Commit a turn after this many quiet chunks following speech.
		end_of_utterance_chunks = max(
			1, int((self.silence_ms / 1000.0) / CHUNK_LENGTH_S)
		)
		min_utterance_chunks = max(
			1, int((MIN_UTTERANCE_MS / 1000.0) / CHUNK_LENGTH_S)
		)

		try:
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

				# ----- Wake-word gating -----
				if not self.awake:
					# Don't send audio to xAI while asleep — only run the detector.
					if self._detect_wake_word(samples):
						await self._wake_up()
						turn_has_audio = False
						silence_chunks = 0
						speech_chunks = 0
					await asyncio.sleep(0)
					continue

				# Track speech activity so we can sleep on prolonged silence.
				if mic_rms >= ENERGY_THRESHOLD or assistant_playing:
					self.last_speech_at = time.monotonic()
				elif (
					self.wake_word_model_path is not None
					and (time.monotonic() - self.last_speech_at) > self.sleep_after_s
				):
					await self._go_to_sleep()
					turn_has_audio = False
					silence_chunks = 0
					speech_chunks = 0
					await asyncio.sleep(0)
					continue

				if self.barge_in_mode == "server":
					# Stream constantly; server_vad handles turn detection + interruption.
					self.vad_active_frames = 0
					await self._send_audio_chunk(audio_bytes)
				else:
					# Local RMS-based turn-taking with echo suppression.
					if assistant_playing:
						if self.force_barge_in:
							self.interrupt_event.set()
							self.force_barge_in = False
							await self._cancel_response()
							await self._send_audio_chunk(audio_bytes)
							turn_has_audio = True
							silence_chunks = 0
							await asyncio.sleep(0)
							continue
						playback_guard = max(
							ENERGY_THRESHOLD, self.recent_playback_rms * ECHO_SUPPRESS_MULTIPLIER
						)
						if mic_rms >= playback_guard:
							self.vad_active_frames += 1
							if self.vad_active_frames >= BARGE_IN_FRAMES_REQUIRED:
								self.interrupt_event.set()
								await self._cancel_response()
								await self._send_audio_chunk(audio_bytes)
								turn_has_audio = True
								silence_chunks = 0
						else:
							self.vad_active_frames = 0
					else:
						self.vad_active_frames = 0
						if mic_rms >= ENERGY_THRESHOLD:
							await self._send_audio_chunk(audio_bytes)
							turn_has_audio = True
							speech_chunks += 1
							silence_chunks = 0
						elif turn_has_audio:
							# Keep streaming trailing silence so we don't clip the
							# last word; commit only after enough quiet AND a
							# minimum amount of actual speech (to avoid spurious
							# commits on stray noise).
							await self._send_audio_chunk(audio_bytes)
							silence_chunks += 1
							if (
								silence_chunks >= end_of_utterance_chunks
								and speech_chunks >= min_utterance_chunks
							):
								await self._commit_and_respond()
								turn_has_audio = False
								silence_chunks = 0
								speech_chunks = 0

				await asyncio.sleep(0)
		except Exception as e:
			print(f"Audio capture error: {e}")
		finally:
			if self.audio_stream and self.audio_stream.active:
				self.audio_stream.stop()
			if self.audio_stream:
				self.audio_stream.close()

	# ---------- Event handling ----------

	async def _on_event(self, event: dict[str, Any]) -> None:
		etype = event.get("type", "")
		try:
			if etype == "session.created" or etype == "session.updated":
				return
			elif etype == "response.created":
				self.active_response_id = (event.get("response") or {}).get("id")
				# If a response races in after an explicit sleep command, cancel it.
				if not self.awake:
					await self._cancel_response()
					return
			elif etype == "response.output_item.added":
				item = event.get("item") or {}
				self.active_item_id = item.get("id", "")
			elif etype == "response.content_part.added":
				self.active_content_index = int(event.get("content_index", 0))
			elif etype == "response.output_audio.delta":
				if not self.awake:
					return
				audio_b64 = event.get("delta") or event.get("audio") or ""
				if not audio_b64:
					return
				pcm = base64.b64decode(audio_b64)
				np_audio = np.frombuffer(pcm, dtype=np.int16)
				item_id = event.get("item_id", self.active_item_id)
				content_index = int(event.get("content_index", self.active_content_index))
				self.output_queue.put_nowait((np_audio, item_id, content_index))
			elif etype == "response.output_audio.done":
				pass
			elif etype == "response.audio_transcript.delta" or etype == "response.output_audio_transcript.delta":
				if not self.awake:
					return
				delta = event.get("delta", "")
				if delta:
					sys.stdout.write(delta)
					sys.stdout.flush()
			elif etype == "response.audio_transcript.done" or etype == "response.output_audio_transcript.done":
				sys.stdout.write("\n")
				sys.stdout.flush()
			elif etype == "conversation.item.input_audio_transcription.completed":
				transcript = event.get("transcript", "")
				if transcript:
					print(f"\n👤 {transcript}")
					if self.awake and self._is_sleep_command(transcript):
						print("🛌 Sleep command detected")
						await self._go_to_sleep()
						return
			elif etype == "input_audio_buffer.speech_started":
				# Server VAD detected user speech; treat as barge-in.
				self.interrupt_event.set()
				await self._cancel_response()
			elif etype == "input_audio_buffer.speech_stopped":
				pass
			elif etype == "response.function_call_arguments.done":
				await self._handle_function_call(event)
			elif etype == "response.done":
				self.active_response_id = None
				usage = ((event.get("response") or {}).get("usage")) or event.get("usage")
				if usage:
					print(
						f"📊 tokens in={usage.get('input_tokens')} out={usage.get('output_tokens')} total={usage.get('total_tokens')}"
					)
			elif etype == "conversation.item.added" or etype == "conversation.created":
				# History bookkeeping — handled via deltas; no-op here.
				pass
			elif etype == "error":
				print(f"Error: {event.get('error', event)}")
			else:
				# Quiet for unknown events; uncomment for debugging.
				# print(f"event: {etype}")
				pass
		except Exception as e:
			print(f"Error processing event {etype}: {_truncate_str(str(e), 200)}")

	async def _handle_function_call(self, event: dict[str, Any]) -> None:
		name = event.get("name", "")
		call_id = event.get("call_id", "")
		raw_args = event.get("arguments", "{}")
		try:
			args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
		except Exception:
			args = {}

		entry = self.tools.get(name)
		if entry is None:
			output = {"error": f"unknown tool: {name}"}
		else:
			_, fn = entry
			try:
				result = fn(**args) if isinstance(args, dict) else fn(args)
				if asyncio.iscoroutine(result):
					result = await result
				output = result if isinstance(result, (dict, list, str, int, float, bool)) else str(result)
				print(f"🛠  {name}({json.dumps(args, ensure_ascii=False)}) -> {_truncate_str(json.dumps(output, ensure_ascii=False), 240)}")
			except Exception as e:
				output = {"error": str(e)}

		await self._send(
			{
				"type": "conversation.item.create",
				"item": {
					"type": "function_call_output",
					"call_id": call_id,
					"output": json.dumps(output),
				},
			}
		)
		# A tool (go_to_sleep) may have ended the conversation — don't kick off a
		# new response if we're now asleep, or Yui would immediately speak again.
		if self.awake:
			await self._send({"type": "response.create"})


async def main():
	args = sys.argv[1:]
	system_prompt: Optional[str] = None
	voice = DEFAULT_VOICE
	model = DEFAULT_MODEL
	barge_in_mode = "rms"
	wake_word_enabled = True
	wake_word_model_path: Optional[str] = DEFAULT_WAKE_WORD_MODEL
	wake_word_threshold = DEFAULT_WAKE_WORD_THRESHOLD
	sleep_after_s = DEFAULT_SLEEP_AFTER_S
	greet_on_wake = True
	memory_path = DEFAULT_MEMORY_PATH
	silence_ms = DEFAULT_SILENCE_MS
	i = 0
	while i < len(args):
		if args[i] == "--system-prompt" and i + 1 < len(args):
			system_prompt = args[i + 1]
			i += 2
		elif args[i] == "--voice" and i + 1 < len(args):
			voice = _normalize_voice(args[i + 1])
			i += 2
		elif args[i] == "--model" and i + 1 < len(args):
			model = args[i + 1]
			i += 2
		elif args[i] == "--barge-in" and i + 1 < len(args):
			barge_in_mode = args[i + 1]
			i += 2
		elif args[i] == "--no-wake-word":
			wake_word_enabled = False
			i += 1
		elif args[i] == "--wake-word-model" and i + 1 < len(args):
			wake_word_model_path = args[i + 1]
			i += 2
		elif args[i] == "--wake-word-threshold" and i + 1 < len(args):
			wake_word_threshold = float(args[i + 1])
			i += 2
		elif args[i] == "--sleep-after" and i + 1 < len(args):
			sleep_after_s = float(args[i + 1])
			i += 2
		elif args[i] == "--no-greet":
			greet_on_wake = False
			i += 1
		elif args[i] == "--memory-path" and i + 1 < len(args):
			memory_path = args[i + 1]
			i += 2
		elif args[i] == "--silence-ms" and i + 1 < len(args):
			silence_ms = int(args[i + 1])
			i += 2
		elif args[i] in ("--help", "-h"):
			print(
				"""
🎤 Yui (Python) — xAI Grok Voice Agent

Usage: python yui.py [options]

Options:
  --system-prompt PROMPT
  --voice VOICE                (Eve | Ara | Leo | Rex | Sal; default Ara)
  --model MODEL                (default grok-voice-think-fast-1.0)
  --barge-in MODE              (rms | server; default rms)
  --no-wake-word               Always-on (skip wake-word gating)
  --wake-word-model PATH       (default ./hey_yoo_wee.onnx)
  --wake-word-threshold FLOAT  (default 0.5)
  --sleep-after SECONDS        Idle seconds before re-arming wake word (default 30)
  --no-greet                   Don't greet on wake; just start listening
  --memory-path PATH           Long-term memory JSON (default ~/.yui/memory.json)
  --silence-ms N               Silence (ms) after speech that ends a turn
                               (default 750; raise if she cuts you off,
                               lower if she feels slow to respond)

Environment:
  XAI_API_KEY                  Required
  GOOGLE_API_KEY               Optional — enables get_weather, get_pollen,
                               get_air_quality (needs Geocoding + the three
                               data APIs enabled in Google Cloud Console)
				"""
			)
			return
		else:
			print(f"❌ Unknown argument: {args[i]}")
			return

	_deps_required()
	instructions = system_prompt or "You are Yui, a friendly voice assistant. Keep replies concise and natural for spoken conversation."
	memory = MemoryStore(memory_path)
	demo = YuiRealtime(
		instructions=instructions,
		memory=memory,
		voice=voice,
		model=model,
		barge_in_mode=barge_in_mode,
		wake_word_model_path=wake_word_model_path if wake_word_enabled else None,
		wake_word_threshold=wake_word_threshold,
		sleep_after_s=sleep_after_s,
		greet_on_wake=greet_on_wake,
		silence_ms=silence_ms,
	)
	await demo.run()


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		print("\nExiting...")
		sys.exit(0)
