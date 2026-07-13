# Spark federated-learning guide

NeuroGrid provides two complementary distributed-learning paths. The online path uses gRPC
between Building, District, and City services. The batch path uses Spark and
TorchDistributor for repeatable training rounds and scalability experiments.

## Execution paths

| Concern | Online gRPC | Spark federation |
| --- | --- | --- |
| Topology | Building, District, and City services | Spark driver and isolated rank processes |
| Transport | gRPC and Protocol Buffers | Spark scheduling, broadcasts, and Parquet |
| Cadence | Continuous ticks | Discrete rounds |
| Recovery | SQLite journals and durable outboxes | Spark task retry and round checkpoints |
| Primary use | Online coordination | Batch training and scalability analysis |

Neither path replaces the other. Both use the same clipping, sparsification, robust
aggregation, payload validation, and model-state conventions.

## Data flow

```text
HEAPO meter files
    |
    v
Spark ETL and data-quality validation
    |
    v
K-Means representative cohort and household-disjoint splits
    |
    +--> optional DDP teacher pretraining
    |
    v
Isolated local training ranks
    |
    v
District Multi-Krum and masked trimmed mean
    |
    v
City weighted aggregation and versioned checkpoint
    |
    v
Parquet metrics and scalability summary
```

## TorchDistributor modes

### Distributed teacher training

`train-spark-ddp` runs conventional `DistributedDataParallel` training. Every rank trains the
same teacher model, gradients are reduced through the `gloo` backend, and the resulting model
can initialize later federated rounds.

Implementation: `src/training/spark_ddp_trainer.py`

### Isolated local training

`simulate-spark-fl` uses TorchDistributor as a Spark-managed process launcher. Ranks do not
initialize a distributed process group and do not exchange gradients during local training.
Each rank trains on its assigned household shard and writes a parameter delta for district
aggregation. This separation preserves federated-learning semantics.

Implementations:

- `src/simulation/spark_fl_runner.py`
- `src/federated/spark_local_train.py`

## Fault tolerance

1. Spark can retry failed rank tasks.
2. Each rank writes its local delta before aggregation.
3. Every round records the input global checkpoint and output metrics.
4. A resumed run validates configuration, schema, manifest, and checkpoint identity.

The local training kernel is deterministic for a fixed seed and input shard, so a retried rank
reproduces the same update.

## Byzantine-resilience experiments

The simulation can designate a bounded fraction of ranks as adversarial and apply scale, sign
flip, or Gaussian-noise attacks. District aggregation uses the same fail-closed Krum and masked
aggregation primitives as the gRPC path. Per-agent decisions and district summaries are written
to Parquet for analysis.

## Reproducibility metadata

Each run records:

- Git commit identifier
- global-checkpoint checksum
- cohort-manifest checksum
- model and feature-schema identifiers
- aggregation and attack configuration
- observed timing and acceptance metrics

Failed benchmark configurations remain visible as failed rows rather than being omitted.

## Commands

```bash
python -m src.cli train-spark-ddp \
  --manifest_path artifacts/sampling/<run_id>/manifest.json

python -m src.cli simulate-spark-fl \
  --manifest_path artifacts/sampling/<run_id>/manifest.json \
  --num_agents 16 \
  --rounds 5

python -m src.cli benchmark-scalability \
  --agent_counts 16 64 256 \
  --include_grpc
```

## Implementation map

| Concern | Module |
| --- | --- |
| Local training kernel | `src/federated/spark_local_train.py` |
| Spark round driver | `src/simulation/spark_fl_runner.py` |
| DDP teacher training | `src/training/spark_ddp_trainer.py` |
| Scalability reporting | `src/reporting/scalability_benchmark.py` |
| Representative sampling | `src/data/representative_sampling.py` |
| Robust aggregation | `src/federated/krum.py`, `aggregation.py`, and `clipping.py` |
