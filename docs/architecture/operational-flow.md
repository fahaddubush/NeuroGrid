# ISMCC Operational Flow

This outlines the complete step-by-step lifecycle of the system.

## The Flow in Plain Language

**Step 1 - Sensors collect everything**
IoT sensors in every building measure temperature, how many people are inside, solar panel output, EV charging state, and grid frequency. This raw data is the starting point of everything.

**Step 2 - The building ISMCC agent wakes up**
Each building has one intelligent agent. When sensors send data, the agent's Sensing (S) module receives it and checks: is this normal? Is anything drifting? It then passes the data to the Memory (M) module, which stores it in three layers - recent events (short-term), daily patterns (long-term), and unusual incidents (episodic memory).

**Step 3 - The building trains its own local model**
The agent's Computation (X) module takes the sensor data + memory context and trains a small local AI model - only on that building's own data. No raw data ever leaves the building. Privacy is protected from the start.

**Step 4 - The building optimizes its energy right now**
While training in the background, the agent also runs the current model to make real-time decisions: pre-cool before peak tariff, delay EV charging to midnight, dim lights in empty rooms. The XAI module writes a simple explanation of every decision so the building manager understands why.

**Step 5 - The building sends model updates upward (FL aggregation - green arrows)**
Instead of sending raw sensor data, the building's Communication (C) module sends only the model gradient - the small update learned from local training - up to the District aggregator. This is federated learning: learning without sharing data.

**Step 6 - The district aggregates building models**
The District ISMCC agent receives model updates from all buildings in the district (e.g. 50 buildings). It runs a Byzantine-resilient aggregation - it checks for any rogue or corrupted updates and filters them out before merging. The result is one district-level model that captures the collective patterns of all buildings.

**Step 7 - The district sends its model to the city**
The district model goes up to the city level (another FL round), following the same privacy-preserving principle. The city receives models from all districts.

**Step 8 - The city builds the global teacher model**
The city's ISMCC agent aggregates all district models into one powerful global city model. This model understands the full picture - grid capacity, renewable energy availability, carbon intensity, weather forecast, demand peaks across the whole city. This is the Teacher.

**Step 9 - The city distills knowledge back down (KD - purple dashed arrows)**
The city teacher compresses its knowledge into smaller, building-type-specific student models: a residential student model, a commercial one, an industrial one. These distilled models are sent back down - first to districts, then to individual buildings. This is Knowledge Distillation.

**Step 10 - Each building receives its updated student model**
The building's Communication (C) module receives the new distilled model and hands it to Computation (X), which hot-swaps it into the inference pipeline immediately. The building is now smarter - it knows city-wide patterns without having seen any other building's data.

**Step 11 - Agents negotiate peer-to-peer energy trading**
Buildings with solar surplus (e.g. Building 2) negotiate directly with buildings in deficit (e.g. Building 4) via the district's P2P trading agent. The Communication module handles the negotiation protocol, and the Computation module decides the bid price using the local optimization model.

**Step 12 - The cycle repeats continuously**
Every hour (or triggered by drift detection), the whole cycle runs again: sense → train locally → aggregate up → distill down → optimize → trade. The system gets smarter over time without ever centralizing raw personal or building data.

## The ISMCC Role in Each Step

| Step | S (Sensing) | M (Memory) | X (Compute) | C (Comm.) |
| --- | --- | --- | --- | --- |
| **Collect data** | reads sensors | stores readings | - | - |
| **Train locally** | detects drift | provides history | trains model | - |
| **Send updates** | - | - | compresses update | sends gradients |
| **Receive model** | - | stores new model | swaps in model | receives from district |
| **Optimize** | reads real-time | retrieves patterns | runs optimizer | sends alerts |
| **P2P trading** | detects surplus | recalls past prices | computes offer | negotiates bids |

## Referenced Architecture Diagrams
- `cognitive_city_ISMCC_full_system_flow.svg`
- `ISMCC_cognitive_agent_internal_architecture.svg`
