"""
X - Computation module (ISMCC Algorithm 2, Diagram 2 lower-left).

Responsibilities (per Diagram 2):
  * Local model f_θ - inference: a_t = π(o_t)
  * Local FL training - mini-batch SGD; produces gradient Δθ
  * Knowledge distillation receive - adapts θ ← distill(W*, θ_local)
  * Energy optimisation solver - converts a forecast into a control action
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from src.data.schema import INPUT_DIM
from src.federated.distillation import DistillationLoss
from src.models.city_lstm import CityLSTM, tier_size

try:
    from ortools.linear_solver import pywraplp
except ImportError:  # OR-Tools is optional - graceful no-op below.
    pywraplp = None


class ComputationModule:
    """Local LSTM + load shedding + KD adapter wrapped for one agent."""

    def __init__(
        self,
        agent_id: str,
        pred_len: int,
        tier: str = "building",
        scaler=None,
    ):
        self.agent_id = agent_id
        self.tier = tier
        self.pred_len = int(pred_len)
        self.scaler = scaler

        sz = tier_size(tier)
        self.model = CityLSTM(
            pred_len=self.pred_len,
            input_dim=INPUT_DIM,
            **sz,
        )
        self.model.eval()
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.distill_loss = DistillationLoss(alpha=0.5, temperature=2.0)

        # 95th-percentile load tracker for adaptive capacity threshold.
        self._kwh_history: list[float] = []
        self._capacity_limit = 1.2

    # ------------------------------------------------------------------ #
    # Local mini-batch SGD step on the most recent STM window.
    # ------------------------------------------------------------------ #
    def train_step(self, stm_window: list[dict], seq_len: int = 24) -> Optional[float]:
        if len(stm_window) < seq_len + 1:
            return None
        feats = np.stack(
            [obs["feature_vector"] for obs in stm_window[-(seq_len + 1) :]]
        )
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)

        x = torch.from_numpy(feats[:-1]).float().unsqueeze(0)
        y_target_raw = feats[-1, 0]
        y = torch.tensor([[float(y_target_raw)]], dtype=torch.float32)

        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x)
        # Only the immediately observed next step is supervised online. A
        # repeated scalar across the full horizon would teach a flat forecast.
        loss = self.criterion(pred[:, :1], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.eval()
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # KD adapt: pull the student toward the city teacher's soft targets.
    # ------------------------------------------------------------------ #
    def distill_from(
        self,
        teacher_model: CityLSTM,
        stm_window: list[dict],
        seq_len: int = 24,
    ) -> Optional[dict]:
        if len(stm_window) < seq_len + 1:
            return None
        feats = np.stack(
            [obs["feature_vector"] for obs in stm_window[-(seq_len + 1) :]]
        )
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)
        x = torch.from_numpy(feats[:-1]).float().unsqueeze(0)
        y = torch.full((1, self.pred_len), float(feats[-1, 0]))

        self.model.train()
        teacher_model.eval()
        with torch.no_grad():
            t_pred = teacher_model(x)
        s_pred = self.model(x)
        losses = self.distill_loss(s_pred, y, t_pred)
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.eval()
        return {k: float(v.item()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #
    # Inference + energy optimisation solver.
    # ------------------------------------------------------------------ #
    def forecast(self, stm_window: list[dict], seq_len: int = 24) -> Optional[np.ndarray]:
        if len(stm_window) < seq_len:
            return None
        feats = np.stack([obs["feature_vector"] for obs in stm_window[-seq_len:]])
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)
        x = torch.from_numpy(feats).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).squeeze(0).cpu().numpy()
        if self.scaler is not None:
            dummy = np.zeros((self.pred_len, INPUT_DIM), dtype=np.float32)
            dummy[:, 0] = pred
            pred = self.scaler.inverse_transform(dummy)[:, 0]
        return pred

    def update_capacity(self, kwh: float) -> None:
        self._kwh_history.append(float(kwh))
        if len(self._kwh_history) >= 96:
            window = self._kwh_history[-7 * 96 :]
            self._capacity_limit = max(0.5, float(np.percentile(window, 95)))

    def optimise_action(self, predicted_kwh: float) -> dict:
        """Return a control packet (action list + reasoning).

        OR-Tools MILP minimises the cost of shedding {EV, HVAC, lighting} loads
        to bring predicted load below the adaptive capacity limit. Falls back
        to a single CRITICAL_SHED action when ortools is unavailable.
        """
        if predicted_kwh is None:
            return {"actions": ["MONITOR"], "reasoning": "STM warming up."}
        if predicted_kwh <= self._capacity_limit:
            return {
                "actions": ["MAINTAIN"],
                "reasoning": (
                    f"Predicted {predicted_kwh:.3f} kWh ≤ adaptive capacity "
                    f"{self._capacity_limit:.3f} kWh."
                ),
            }
        required = predicted_kwh - self._capacity_limit
        if pywraplp is None:
            return {
                "actions": ["CRITICAL_SHED"],
                "reasoning": "OR-Tools unavailable; emergency total shed.",
            }
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return {"actions": ["CRITICAL_SHED"], "reasoning": "Solver init failed."}
        s_ev = solver.NumVar(0.0, 0.3 * predicted_kwh, "s_ev")
        s_hvac = solver.NumVar(0.0, 0.4 * predicted_kwh, "s_hvac")
        s_light = solver.NumVar(0.0, 0.2 * predicted_kwh, "s_light")
        solver.Add(s_ev + s_hvac + s_light >= required)
        solver.Minimize(1.0 * s_ev + 3.0 * s_hvac + 10.0 * s_light)
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return {"actions": ["CRITICAL_SHED"], "reasoning": "Solver infeasible."}
        actions: list[str] = []
        details: list[str] = []
        if s_ev.solution_value() > 0.01:
            actions.append("SHED_EV")
            details.append(f"defer {s_ev.solution_value():.3f} kWh EV charging")
        if s_hvac.solution_value() > 0.01:
            actions.append("CURTAIL_HVAC")
            details.append(f"curtail {s_hvac.solution_value():.3f} kWh HVAC")
        if s_light.solution_value() > 0.01:
            actions.append("DIM_LIGHTING")
            details.append(f"dim lighting by {s_light.solution_value():.3f} kWh")
        if not actions:
            actions = ["MAINTAIN"]
            details = ["Solver returned no positive shed."]
        return {
            "actions": actions,
            "reasoning": (
                f"Predicted {predicted_kwh:.3f} kWh > capacity "
                f"{self._capacity_limit:.3f} kWh. Plan: " + "; ".join(details)
            ),
        }

    # ------------------------------------------------------------------ #
    # Next-day scheduler.
    # Takes a full forecast vector (e.g. 96 steps = 24 h at 15 min) and
    # solves an LP that minimises tariff cost across:
    #   - battery charge/discharge (with SoC bounds)
    #   - HVAC pre-cool window (shift k slots earlier)
    #   - peak shaving (cap drawn from grid at peak slots)
    # Inputs are dimensionless slots; the caller assembles them.
    # ------------------------------------------------------------------ #
    def optimise_schedule(
        self,
        forecast_kwh: np.ndarray,
        tariff: Optional[np.ndarray] = None,
        battery_capacity_kwh: float = 10.0,
        battery_max_rate_kwh: float = 2.0,
        battery_initial_soc: float = 0.5,
        peak_cap_kwh: Optional[float] = None,
        battery_efficiency: float = 0.95,
    ) -> dict:
        """Plan a next-day schedule. Returns charge/discharge per slot,
        recommended pre-cool window, expected peak window, and cost saved.

        Uses OR-Tools GLOP. When OR-Tools is unavailable it returns a safe
        no-op plan rather than claiming savings without enforcing constraints.
        """
        forecast = np.asarray(forecast_kwh, dtype=np.float64).reshape(-1)
        n = len(forecast)
        if n == 0:
            return {"status": "empty_forecast"}
        if not np.isfinite(forecast).all() or np.any(forecast < 0):
            raise ValueError("forecast must contain finite, non-negative values.")
        if battery_capacity_kwh <= 0 or battery_max_rate_kwh <= 0:
            raise ValueError("battery capacity and rate must be positive.")
        if not 0.0 <= battery_initial_soc <= 1.0:
            raise ValueError("battery_initial_soc must be in [0, 1].")
        if not 0.0 < battery_efficiency <= 1.0:
            raise ValueError("battery_efficiency must be in (0, 1].")

        # Default tariff: simple 3-band TOU (off-peak / mid / peak) repeated
        # to fit `n` slots. Off-peak 0-6 + 22-24, peak 17-21, mid otherwise.
        if tariff is None:
            slots_per_hour = max(1, n // 24)
            tariff = np.empty(n, dtype=np.float64)
            for i in range(n):
                hour = (i // slots_per_hour) % 24
                if hour < 6 or hour >= 22:
                    tariff[i] = 0.10
                elif 17 <= hour < 21:
                    tariff[i] = 0.32
                else:
                    tariff[i] = 0.20
        tariff = np.asarray(tariff, dtype=np.float64).reshape(-1)
        if len(tariff) != n:
            raise ValueError("tariff length must match forecast length.")
        if not np.isfinite(tariff).all() or np.any(tariff < 0):
            raise ValueError("tariff must contain finite, non-negative values.")

        baseline_cost = float(np.sum(forecast * tariff))
        peak_idx = int(np.argmax(forecast))

        if pywraplp is None:
            # Safe fallback: do not invent a schedule without a solver capable
            # of enforcing state-of-charge and terminal-energy constraints.
            charge = np.zeros(n)
            discharge = np.zeros(n)
            return {
                "status": "solver_unavailable",
                "baseline_cost": baseline_cost,
                "scheduled_cost": baseline_cost,
                "savings": 0.0,
                "battery_charge_kwh": charge.tolist(),
                "battery_discharge_kwh": discharge.tolist(),
                "peak_slot": peak_idx,
                "precool_start_slot": max(0, peak_idx - 4),
                "tariff": tariff.tolist(),
            }

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return {"status": "solver_init_failed", "baseline_cost": baseline_cost}

        # Decision vars per slot: battery charge and discharge. Load cannot be
        # removed by an unconstrained synthetic "peak shave" variable.
        c = [solver.NumVar(0.0, battery_max_rate_kwh, f"c_{i}") for i in range(n)]
        d = [solver.NumVar(0.0, min(battery_max_rate_kwh, forecast[i]), f"d_{i}") for i in range(n)]

        # SoC trajectory: monotone running sum, must stay in [0, capacity].
        soc0 = battery_initial_soc * battery_capacity_kwh
        for i in range(n):
            running = (
                soc0
                + battery_efficiency * sum(c[: i + 1])
                - sum(d[: i + 1]) / battery_efficiency
            )
            solver.Add(running >= 0.0)
            solver.Add(running <= battery_capacity_kwh)
        solver.Add(
            soc0
            + battery_efficiency * sum(c)
            - sum(d) / battery_efficiency
            == soc0
        )

        # Optional hard peak cap.
        if peak_cap_kwh is not None:
            for i in range(n):
                solver.Add(forecast[i] - d[i] + c[i] <= peak_cap_kwh)

        # Objective: total tariff cost on physically conserved net grid draw.
        cost_terms = []
        for i in range(n):
            net = forecast[i] - d[i] + c[i]
            cost_terms.append(tariff[i] * net)
        solver.Minimize(solver.Sum(cost_terms))

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return {"status": "infeasible", "baseline_cost": baseline_cost}

        charge = np.array([c[i].solution_value() for i in range(n)])
        discharge = np.array([d[i].solution_value() for i in range(n)])
        net = forecast - discharge + charge
        scheduled_cost = float(np.sum(net * tariff))

        return {
            "status": "optimal",
            "baseline_cost": baseline_cost,
            "scheduled_cost": scheduled_cost,
            "savings": baseline_cost - scheduled_cost,
            "battery_charge_kwh": charge.tolist(),
            "battery_discharge_kwh": discharge.tolist(),
            "peak_slot": peak_idx,
            "precool_start_slot": max(0, peak_idx - 4),
            "tariff": tariff.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Federation hooks.
    # ------------------------------------------------------------------ #
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def parameter_delta(self, baseline_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Δθ = θ_local - θ_baseline. Used by the building uplink (TopK)."""
        local = self.snapshot()
        delta: dict[str, torch.Tensor] = {}
        for k, v in local.items():
            base = baseline_state.get(k, torch.zeros_like(v))
            delta[k] = (v - base).cpu()
        return delta

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Deep copy of current parameters (for Δθ computation later)."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def adopt_global(self, global_state: dict[str, torch.Tensor]) -> None:
        """Replace local weights with the broadcast global model W*."""
        # Tolerate missing keys gracefully - student may be smaller than teacher.
        own = self.model.state_dict()
        for k, v in global_state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        self.model.load_state_dict(own)
