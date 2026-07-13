"""
Energy recommendation layer - converts (forecast + scheduler output) into a
human-readable suggestion string.

Uses a local Ollama LLM API to generate natural-language energy advice.
The agent core invokes it when the building agent was constructed with
`enable_recommendations=True`. Otherwise the agent path is unchanged.

Fallback behaviour: if Ollama is not running OR the API call fails, returns
a deterministic template string.
"""
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RecommendationResult:
    text: str
    source: str  # "ollama" | "template" | "fallback_error"
    peak_slot: int
    precool_slot: int
    estimated_savings: float


_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_SYSTEM_PROMPT = (
    "You are an energy advisor for a smart-home federated learning agent. "
    "Given a 24-hour forecast in 15-minute steps, a tariff schedule, a "
    "recommended battery plan, and a peak window, produce a short (3-4 "
    "sentences) plain-English recommendation for the homeowner. Mention: "
    "(1) when usage will peak, (2) when to pre-cool / pre-heat the HVAC, "
    "(3) battery charge/discharge windows, (4) when to schedule heavy appliances "
    "(like washing machines, dishwashers, and EV charging) to avoid the peak. "
    "CRITICAL: Never mention raw 'slot' numbers (like slot 72). Always translate "
    "slots into clock times (e.g., 6:00 PM). Do not mention kWh totals."
)


def _format_template(forecast: np.ndarray, schedule: dict) -> str:
    """Deterministic fallback when no LLM is available."""
    n = len(forecast)
    slots_per_hour = max(1, n // 24)
    default_peak = int(np.argmax(forecast)) if n else 0
    peak = int(schedule.get("peak_slot", default_peak))
    precool = int(schedule.get("precool_start_slot", max(0, peak - 4)))

    def slot_to_clock(s: int) -> str:
        h = (s // slots_per_hour) % 24
        m = (s % slots_per_hour) * (60 // slots_per_hour)
        return f"{h:02d}:{m:02d}"

    charge = np.asarray(schedule.get("battery_charge_kwh", []), dtype=float)
    discharge = np.asarray(schedule.get("battery_discharge_kwh", []), dtype=float)
    charge_window = (
        slot_to_clock(int(np.argmax(charge))) if charge.size and charge.max() > 0 else "off-peak"
    )
    discharge_window = (
        slot_to_clock(int(np.argmax(discharge)))
        if discharge.size and discharge.max() > 0
        else slot_to_clock(peak)
    )
    savings = float(schedule.get("savings", 0.0))

    return (
        f"Tomorrow's usage is expected to peak around {slot_to_clock(peak)}. "
        f"Pre-cool your HVAC starting around {slot_to_clock(precool)}. "
        f"Charge the battery near {charge_window} and discharge near {discharge_window}. "
        f"Run your washing machine and other heavy appliances during the {charge_window} window to save costs."
    )


class EnergyRecommender:
    """Wraps local Ollama API + template fallback."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 240,
        timeout_seconds: float = 5.0,
    ):
        # api_key is retained for backward compatibility with the former
        # hosted-provider interface; local Ollama does not use it.
        _ = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        parsed = urllib.parse.urlparse(self.ollama_url)
        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            if os.getenv("NEUROGRID_ALLOW_REMOTE_LLM", "0") != "1":
                raise RuntimeError(
                    "Refusing to send household data to a non-loopback LLM endpoint. "
                    "Set NEUROGRID_ALLOW_REMOTE_LLM=1 only after reviewing privacy controls."
                )
            if parsed.scheme != "https":
                raise RuntimeError("Remote LLM endpoints must use HTTPS.")
        
    @property
    def has_llm(self) -> bool:
        return True # Always attempt Ollama locally

    def recommend(
        self,
        forecast: np.ndarray,
        schedule: dict,
        household_id: Optional[str] = None,
    ) -> RecommendationResult:
        forecast = np.asarray(forecast, dtype=float).reshape(-1)
        peak_slot = int(schedule.get("peak_slot", int(np.argmax(forecast)) if forecast.size else 0))
        precool_slot = int(schedule.get("precool_start_slot", max(0, peak_slot - 4)))
        savings = float(schedule.get("savings", 0.0))

        def slot_to_clock(s: int) -> str:
            h = (int(s) // 4) % 24
            m = (int(s) % 4) * 15
            ampm = "AM" if h < 12 else "PM"
            h_disp = h if 0 < h <= 12 else (h - 12 if h > 12 else 12)
            return f"{h_disp}:{m:02d} {ampm}"

        peak_time = slot_to_clock(peak_slot)
        precool_time = slot_to_clock(precool_slot)

        user_msg = (
            "Household: local-anonymous-home\n"
            f"Forecast (kWh per 15-min slot): {[round(float(v), 3) for v in forecast[:96]]}\n"
            f"Peak Time: {peak_time}\n"
            f"Pre-cool HVAC Start Time: {precool_time}\n"
            f"Battery Charge Plan: {[round(float(v), 3) for v in schedule.get('battery_charge_kwh', [])[:96]]}\n"
            f"Battery Discharge Plan: {[round(float(v), 3) for v in schedule.get('battery_discharge_kwh', [])[:96]]}\n"
            f"Tariff status: {schedule.get('status', 'unknown')}\n"
            f"Estimated savings: {savings:.3f}\n"
            "Write the recommendation using only clock times."
        )
        
        try:
            data = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                "stream": False,
                "options": {"num_predict": int(self.max_tokens), "temperature": 0.2},
            }).encode("utf-8")
            
            req = urllib.request.Request(self.ollama_url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result.get("message", {}).get("content", "").strip()
                
            # Fail closed when the model ignores the grounded clock times or
            # exposes internal slot indices. Templates are safer than plausible
            # but ungrounded advice.
            normalized = re.sub(r"\s+", " ", text).strip().lower()
            grounded = peak_time.lower() in normalized and precool_time.lower() in normalized
            if not text or "slot " in normalized or not grounded or len(text) > 1500:
                text = _format_template(forecast, schedule)
                source = "fallback_error"
            else:
                source = "ollama"
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as e:
            logging.warning("Ollama recommendation unavailable: %s", e)
            text = _format_template(forecast, schedule)
            source = "fallback_error"

        return RecommendationResult(
            text=text,
            source=source,
            peak_slot=peak_slot,
            precool_slot=precool_slot,
            estimated_savings=savings,
        )
