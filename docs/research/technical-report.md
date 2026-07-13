# NeuroGrid technical report

## Abstract

NeuroGrid combines distributed data processing, local sequence forecasting, hierarchical
federated learning, and durable multi-agent coordination for smart-grid research. PySpark
transforms high-frequency meter readings into versioned Parquet datasets. Building agents train
local CityLSTM models, District services filter and aggregate bounded model updates, and the City
service maintains a versioned global model. The design emphasizes causal features, explicit
failure recovery, robust aggregation, and clear security boundaries.

## Motivation

Household consumption is non-stationary, strongly seasonal, and sensitive to local behavior.
Centralizing raw telemetry creates privacy, bandwidth, and operational risks. NeuroGrid explores
an alternative in which each building retains its measurements, learns locally, and exchanges
validated model updates through a hierarchical coordination layer.

## Data pipeline

The Spark pipeline processes 15-minute meter data, validates household and weather identifiers,
applies causal missing-value handling, produces lag and rolling features, and writes an immutable
run directory in Parquet format. Publication uses an atomic `CURRENT` pointer so an incomplete
run cannot replace the last valid dataset.

Representative sampling uses Spark MLlib K-Means over household-level statistics. A fixed seed
and cluster-stratified selection produce a reproducible cohort while reducing the cost of local
experiments. Household-disjoint train, validation, and test assignments prevent identity leakage
across evaluation splits.

## ISMCC architecture

ISMCC means **Integrated Sensing, Memory, Computation, and Communication**. NeuroGrid applies
this model at three levels:

- **Building:** senses meter and context features, retains bounded local memory, forecasts demand,
  optimizes a battery schedule, and sends compressed model updates.
- **District:** receives concurrent uploads, persists them transactionally, rejects invalid or
  Byzantine updates, and forwards a durable district summary.
- **City:** aggregates district summaries, evaluates candidate updates, records convergence state,
  and publishes versioned global checkpoints.

## Learning and aggregation

The official daily forecasting configuration maps 96 historical 15-minute intervals to the next
96 intervals. Local training uses gradient clipping and finite-value checks. Optional Top-K
compression includes coordinate masks and error feedback, so omitted values are not interpreted
as zeros and residual information can be transmitted in later rounds.

District aggregation validates Krum's minimum cohort requirement before selection. Surviving
updates are combined with coordinate-aware masked aggregation. City aggregation weights accepted
district summaries by contributing sample count and rejects candidates that violate configured
update limits.

## Reliability and security

District and City state is stored in SQLite with write-ahead logging and transactional state
transitions. Durable outboxes and idempotency keys allow interrupted uploads to resume safely.
Federated payloads are size-limited, schema-hashed, and loaded through PyTorch's restricted
deserializer.

Local development uses loopback-only insecure gRPC. Network deployment requires mutual TLS and
agent identities bound to certificate subjects. These mechanisms authenticate transport but do
not provide cryptographic secure aggregation; District services can inspect individual updates.

## Evaluation scope

The repository includes numerical, persistence, concurrency, transport, ETL, recommender, and
integration tests. Benchmark commands emit observed results and retain failed configurations.
Claims about forecasting accuracy, throughput, privacy budgets, or adversarial tolerance require
results from the target dataset and deployment environment; they are not inferred from unit tests.

## References

1. McMahan, H. B., et al. (2017). Communication-Efficient Learning of Deep Networks from
   Decentralized Data. AISTATS.
2. Blanchard, P., et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient
   Descent. NeurIPS.
3. Zaharia, M., et al. (2016). Apache Spark: A Unified Engine for Big Data Processing.
   Communications of the ACM, 59(11), 56-65.
