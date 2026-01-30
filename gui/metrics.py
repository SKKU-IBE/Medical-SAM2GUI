"""Lightweight usage metrics recorder for usability evaluation.

This helper keeps per-session events, stage timings, and counters, and writes
JSONL rows under a timestamped file in `logs/` by default.
"""
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class UsageMetricsRecorder:
    """Simple, thread-safe recorder for GUI usage metrics."""

    def __init__(self, log_dir: Optional[str] = None, json_indent: int = 2):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"usage_metrics_{ts}.jsonl"
        self._session: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self.json_indent = json_indent

    def start_session(self, context: Dict[str, Any]) -> str:
        """Start a new session; context can include patient/mode metadata."""
        session_id = str(uuid.uuid4())
        self._session = {
            "session_id": session_id,
            "context": context,
            "started_at": time.time(),
            "events": [],
            "stages": [],
            "counters": {},
            "info": {},
        }
        return session_id

    def is_active(self) -> bool:
        return self._session is not None

    def add_event(self, name: str, **data: Any) -> None:
        if not self._session:
            return
        self._session["events"].append({"ts": time.time(), "name": name, **data})

    def record_stage(self, name: str, start_ts: float, end_ts: Optional[float] = None, **data: Any) -> None:
        if not self._session:
            return
        end_ts = end_ts or time.time()
        self._session["stages"].append(
            {
                "name": name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_sec": round(end_ts - start_ts, 4),
                **data,
            }
        )

    def inc_counter(self, name: str, inc: int = 1) -> None:
        if not self._session:
            return
        counters = self._session["counters"]
        counters[name] = counters.get(name, 0) + inc

    def set_info(self, key: str, value: Any) -> None:
        if not self._session:
            return
        self._session["info"][key] = value

    def finalize(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._session:
            return
        payload = dict(self._session)
        payload["finished_at"] = time.time()
        if extra:
            payload["extra"] = extra
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, indent=self.json_indent) + "\n")
        self._session = None

