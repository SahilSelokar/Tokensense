import json
import logging
import sqlite3
import urllib.request
from typing import List, Dict, Any, Optional

from tokensense.events import CallEvent

class BaseOutput:
    def write(self, event: CallEvent) -> None:
        raise NotImplementedError("Subclasses must implement write")

class Stdout(BaseOutput):
    def write(self, event: CallEvent) -> None:
        status_indicator = "✓" if event.error is None else "✗"
        meta_str = (
            f"model={event.model} | "
            f"in={event.input_tokens} out={event.output_tokens} tokens | "
            f"${event.cost_usd:.4f} | "
            f"{int(event.latency_ms)}ms"
        )
        if event.error:
            meta_str += f" | error={event.error}"
            
        print(f"→ {status_indicator} {meta_str}")
        
        if event.prompt:
            print(f"  [Prompt] {event.prompt}")
        if event.response:
            print(f"  [Response] {event.response}")

class SQLite(BaseOutput):
    def __init__(self, db_path: str = "./tokensense.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                latency_ms INTEGER,
                user_id TEXT,
                session_id TEXT,
                tags TEXT,
                routed_tier TEXT,
                error TEXT
            )
        """)
        conn.commit()
        conn.close()

    def write(self, event: CallEvent) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO calls (
                ts, model, provider, input_tokens, output_tokens, cost_usd, 
                latency_ms, user_id, session_id, tags, routed_tier, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.ts,
            event.model,
            event.provider,
            event.input_tokens,
            event.output_tokens,
            event.cost_usd,
            event.latency_ms,
            event.user_id,
            event.session_id,
            json.dumps(event.tags) if event.tags is not None else None,
            event.routed_tier,
            event.error
        ))
        conn.commit()
        conn.close()

class Logger(BaseOutput):
    def __init__(self, logger_name: str = "tokensense"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def write(self, event: CallEvent) -> None:
        # We don't include prompt/response in the main logger by default to keep json clean,
        # but to_dict() includes them if they were explicitly enabled.
        log_message = json.dumps(event.to_dict())
        if event.error is None:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)

class HTTP(BaseOutput):
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def write(self, event: CallEvent) -> None:
        req_data = json.dumps(event.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=req_data,
            headers=self.headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
        except Exception:
            pass

class Multi(BaseOutput):
    def __init__(self, *outputs: BaseOutput):
        self.outputs = list(outputs)

    def write(self, event: CallEvent) -> None:
        for output in self.outputs:
            try:
                output.write(event)
            except Exception:
                pass
