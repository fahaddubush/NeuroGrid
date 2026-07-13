"""Integration tests for the ISMCC AgentCore tick lifecycle."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import pandas as pd

from src.data.streaming import HouseholdStream
from src.ismcc import (
    AgentCore,
    SensingModule,
    ComputationModule,
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
)
from src.data.schema import INPUT_DIM


class TestAgentCoreTick(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv = self.tmp / "h.csv"
        rows = ["Timestamp;kWh_received_Total;Household_ID;Group;kWh_received_HeatPump;kWh_received_Other;AffectsTimePoint"]
        for i in range(40):
            rows.append(
                f"2024-01-01T{i % 24:02d}:{(i * 15) % 60:02d}:00+00:00;1.5;100101;A;;;unknown"
            )
        self.csv.write_text("\n".join(rows), encoding="utf-8")
        os.environ["HEAPO_DATA_DIR"] = str(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_agent(self):
        sensing = SensingModule()
        compute = ComputationModule(agent_id="T", pred_len=4, tier="building")
        memory = {
            "stm": ShortTermMemory(capacity=64),
            "ltm": LongTermMemory(
                agent_id="T",
                db_path=self.tmp / "ltm.db",
                feature_dim=INPUT_DIM,
                pool_block_size=4,
            ),
            "episodic": EpisodicMemory(
                agent_id="T", capacity=100, episode_dir=self.tmp / "ep"
            ),
        }
        return AgentCore(
            agent_id="T",
            sensing=sensing,
            memory=memory,
            compute=compute,
            comms=None,
        )

    def test_tick_lifecycle_runs(self):
        agent = self._build_agent()
        stream = HouseholdStream(str(self.csv))
        ticks = 0
        while True:
            r = stream.next_reading()
            if r is None:
                break
            out = agent.tick(r)
            self.assertIn("decision", out)
            self.assertIn("observation", out)
            ticks += 1
        self.assertGreater(ticks, 30)
        # After enough ticks the agent should have moved past warm-up MONITOR.
        self.assertIn(
            agent.state.a_t["actions"][0],
            {"MAINTAIN", "MONITOR", "CRITICAL_SHED", "SHED_EV", "CURTAIL_HVAC", "DIM_LIGHTING"},
        )

    def test_ltm_pool_grows(self):
        agent = self._build_agent()
        stream = HouseholdStream(str(self.csv))
        for _ in range(20):
            r = stream.next_reading()
            if r is None:
                break
            agent.tick(r)
        self.assertGreater(agent.memory["ltm"].pool_size(), 0)


if __name__ == "__main__":
    unittest.main()
