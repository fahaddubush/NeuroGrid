# ISMCC Scale Across System Tiers

The ISMCC architecture scales hierarchically across three tiers: Building, District, and City. Each tier instantiates an ISMCC agent specialized for its scope.

| Tier | Sensing (S) | Memory (M) | Computation (X) | Communication (C) |
| --- | --- | --- | --- | --- |
| **Building** | Zone-level IoT, occupancy | Daily patterns, appliance states | Real-time setpoint optimization | FL gradients, P2P bids |
| **District** | Aggregated virtual sensors, grid signals | District consumption history, energy contracts | Demand forecasting, P2P clearing | FL aggregation, Byzantine detection |
| **City** | Digital twin state, carbon intensity, weather | City-wide patterns, policy memory, regulatory constraints | City-scale optimization, teacher model update | Global FL, KD distribution, grid operator API |

## Research contribution
"We propose ISMCC-MAS (Integrated Sensing, Memory, Computation, and Communication Multi-Agent System) for cognitive city energy management, in which each agent at every tier of the FL-KD hierarchy instantiates a full ISMCC unit. Unlike prior work that treats FL participants as passive gradient sources, our agents exhibit memory-augmented reasoning, selective communication, and drift-aware distillation requests - enabling the system to adapt to temporal distribution shifts, coordinate P2P energy trading, and provide XAI justifications, all within a privacy-preserving federated framework."
