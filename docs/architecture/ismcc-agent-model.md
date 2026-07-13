# The ISMCC-Agent Model

The key idea is that every agent in the hierarchy is not just a decision node - it is a complete ISMCC unit. Each agent $a_i$ is formally defined as:
$a_i = \langle S_i, M_i, X_i, C_i \rangle$
and the four modules are tightly coupled, not isolated silos. This coupling is what makes it "intelligent" rather than just automated.

## Module-by-Module Design for Energy Management

### Sensing (S) - Beyond Raw IoT
Traditional IoT just reads sensors. Here, S performs active, purposeful sensing:
- **Physical sensing**: Temperature, occupancy, solar irradiance, EV state-of-charge, grid frequency.
- **Virtual sensing**: Infers missing values from neighboring agents (if sensor fails, query the shared memory pool of nearby buildings).
- **Drift detection**: Monitors when the local data distribution shifts (seasonal change, new occupant behavior), which triggers a flag to C to request a fresh KD update from the city model.
- **Context binding**: Tags every reading with time-of-day, day-type (weekday/Ramadan/holiday), and grid carbon intensity - this enriched context flows to M and X.

### Memory (M) - The Cognitive Backbone
This is what separates an ISMCC agent from a plain IoT node:
- **Short-term memory (STM)**: Last 15-minute window of readings - feeds real-time inference in X.
- **Long-term memory (LTM)**: Compressed seasonal consumption profiles - used by X for 24h-ahead forecasting.
- **Episodic memory**: Logs of past anomalies, demand spikes, successful P2P trades - agents learn from their own history.
- **Shared knowledge pool**: A distributed memory shared across peer agents at the same tier (buildings in the same district share occupancy patterns and tariff structures without sharing raw data - privacy-preserving).
- **Distilled model store**: Holds the current student model received from the teacher, version-tagged so X knows when it's stale.

### Computation (X) - The Reasoning Engine
X is where the agent "thinks":
- **Inference**: Runs the student model (distilled from the city teacher) to predict optimal HVAC/lighting setpoints.
- **FL training**: Fine-tunes the local model on fresh sensor data and pushes gradients to C.
- **Agentic planning**: An LLM-based or rule-based planner that composes multi-step actions ("if peak tariff starts in 20 min AND battery at 80% → pre-cool the building now").
- **Load optimization**: Solves a short-horizon scheduling problem (MILP or RL-based) using STM from M.
- **XAI explanation**: Generates human-readable justifications for every action, stored in M's episodic log and surfaced to building managers.

### Communication (C) - Protocol-Aware and Selective
C does not broadcast everything - it is compute-aware and bandwidth-aware:
- **FL uplink**: Sends local model gradients (not raw data) to the district aggregator.
- **KD downlink**: Receives compressed student models from the district/city teacher.
- **P2P energy trading**: Negotiates with neighboring building agents on energy surplus/deficit (this is a new agent-to-agent communication channel beyond standard FL).
- **Alert propagation**: If S detects a cyberphysical anomaly, C bypasses the FL cycle and sends a direct alert upward.
- **Selective sharing**: C decides what to communicate based on X's computation budget and network congestion signals from the 5G/edge mesh.

## The Four Internal Couplings (The Novelty)
The arrows in the diagram represent four functional couplings that define the system's intelligence:
1. **S → M (Data)**: Every sensed reading is stored with its context into M. Memory is not a passive log - S decides relevance before writing (only anomalies and distribution shifts always write; normal readings are summarized).
2. **M → X (Context)**: Before X runs inference or optimization, it queries M for historical context. This is analogous to RAG for LLMs - the agent grounds its decisions in memory.
3. **S → C (Events)**: Drift or anomaly events from S directly trigger C to communicate upward, bypassing the scheduled FL cycle. This is the reactive path.
4. **C → X (Models)**: When C receives a new distilled student model from the teacher, it hands it to X which hot-swaps it into the inference pipeline - no restart, no downtime.
