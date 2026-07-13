"""
Single command-line entry point for the NeuroGrid project.

Subcommands:
    etl                Spark ETL: HEAPO CSVs → Parquet (gRPC path input)
    etl-spark-fl       Spark ETL → dedicated Parquet root for the Spark FL path
    sample-households  Spark MLlib representative cohort selection
    train              Train one stage of the curriculum
    curriculum         Run the full 6-stage gradual forecasting pipeline
    train-spark-ddp    TorchDistributor + DDP teacher pretraining
    evaluate           Evaluate a trained run bundle
    simulate           Spawn the multi-process 3-tier gRPC simulation
    simulate-spark-fl  Spark + TorchDistributor federated-learning rounds
    benchmark-summary  Roll up ETL / sampling / training / simulation evidence
    benchmark-scalability  Sweep agent counts; compare gRPC vs Spark FL paths
"""
from __future__ import annotations

import os
import sys

# Windows-specific environment variables for PyTorch distributed and Spark stability.
# Must be set before torch or pyspark are imported anywhere.
os.environ["USE_LIBUV"] = "0"
os.environ["TP_SOCKET_IFNAME"] = "lo"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _cmd_etl(args):
    from src.data.spark_etl import run_etl
    run_etl(
        max_households=getattr(args, "max_households", None),
        household_manifest=getattr(args, "household_manifest", None),
        source_timezone=getattr(args, "source_timezone", None),
    )


def _cmd_etl_spark_fl(args):
    """Run ETL into the dedicated Spark-FL Parquet root.

    Routes the same Spark ETL pipeline to a separate output directory so the
    Spark FL implementation has its own, isolated data store. This keeps the
    gRPC path's `processed_parquet_15min/` untouched and makes the two paths
    independently auditable / reproducible.
    """
    from src.data.spark_etl import run_etl
    from src.simulation.spark_fl_runner import SPARK_FL_PARQUET_ROOT
    output_dir = args.output_dir or SPARK_FL_PARQUET_ROOT
    run_etl(
        output_dir=output_dir,
        max_households=getattr(args, "max_households", None),
        household_manifest=getattr(args, "household_manifest", None),
        source_timezone=getattr(args, "source_timezone", None),
    )
    print(f"[etl-spark-fl] Spark-FL Parquet store ready at: {output_dir}")


def _cmd_train_spark_ddp(args):
    """TorchDistributor + DDP city-teacher pretraining."""
    from src.training.spark_ddp_trainer import DDPTrainConfig, train_city_teacher_ddp
    cfg = DDPTrainConfig(
        parquet_root=args.parquet_root,
        output_dir=args.output_dir,
        tier=args.tier,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        world_size=args.world_size,
        seed=args.seed,
        manifest_path=args.manifest_path,
    )
    out = train_city_teacher_ddp(cfg)
    print(f"[train-spark-ddp] Teacher bundle: {out}")


def _cmd_simulate_spark_fl(args):
    """Spark + TorchDistributor federated-learning rounds."""
    from src.simulation.spark_fl_runner import SparkFLConfig, run_spark_fl
    cfg = SparkFLConfig(
        num_agents=args.num_agents,
        num_districts=args.num_districts,
        rounds=args.rounds,
        f_byzantine=args.f_byzantine,
        byzantine_fraction=args.byzantine_fraction,
        byzantine_attack=args.byzantine_attack,
        clip_threshold=args.clip_threshold,
        trim_ratio=args.trim_ratio,
        convergence_epsilon=args.convergence_epsilon,
        convergence_patience=args.convergence_patience,
        spark_master=args.spark_master,
        resume_from_round=args.resume_from_round,
        parquet_root=args.parquet_root,
        run_root=args.run_root,
        local_steps_building=args.local_steps_building,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        tier=args.tier,
        lr=args.lr,
        batch_size=args.batch_size,
        dp_sigma=args.dp_sigma,
        dp_clip_C=args.dp_clip_C,
        topk_ratio=args.topk_ratio,
        seed=args.seed,
        manifest_path=args.manifest_path,
    )
    run_dir = run_spark_fl(cfg)
    print(f"[simulate-spark-fl] Run dir: {run_dir}")


def _cmd_benchmark_scalability(args):
    """Sweep agent counts; compare gRPC vs Spark FL paths."""
    from src.reporting.scalability_benchmark import run_scalability_sweep
    out_path = run_scalability_sweep(
        agent_counts=args.agent_counts,
        rounds=args.rounds,
        num_districts=args.num_districts,
        local_steps_building=args.local_steps_building,
        parquet_root=args.parquet_root,
        output_root=args.output_root,
        include_grpc=args.include_grpc,
    )
    print(f"[benchmark-scalability] Wrote: {out_path}")


def _cmd_sample_households(args):
    """Run KMeans-stratified representative-household sampling."""
    from src.data.representative_sampling import run_sampling
    run_dir, selected = run_sampling(
        target_n=args.target_n,
        k_clusters=args.k,
        seed=args.seed,
        data_dir=args.data_dir,
        output_root=args.output_root,
    )
    print(f"[sample-households] Selected {len(selected)} households.")
    print(f"[sample-households] Manifest: {run_dir / 'manifest.json'}")
    print(f"[sample-households] Summary : {run_dir / 'summary.md'}")


def _cmd_forecast_daily(args):
    """Train a 24-hour next-day forecaster on the
    15-min CSV data. Uses pred_len=96 (24 h × 4 slots/h). Warm-starts from
    a curriculum h3h checkpoint if one exists."""
    from src.data.schema import DAILY_HORIZON
    from src.training.trainer import train_stage
    train_stage(
        stage=DAILY_HORIZON,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_households=args.max_households,
        patience=getattr(args, "patience", 3),
        pretrained_path=args.pretrained,
        tier=args.tier,
        lr=getattr(args, "lr", 5e-4),
        dropout=getattr(args, "dropout", None),
        use_weather=not args.no_weather,
        manifest_path=getattr(args, "manifest_path", None),
        seed=getattr(args, "seed", 0),
        loss=getattr(args, "loss", "peak_weighted_smooth_l1"),
        peak_weight=getattr(args, "peak_weight", 4.0),
        peak_quantile=getattr(args, "peak_quantile", 0.95),
        peak_lambda=getattr(args, "peak_lambda", 0.25),
    )


def _cmd_train(args):
    from src.training.trainer import train_stage
    from src.data.schema import CURRICULUM_STAGES
    stage = next((s for s in CURRICULUM_STAGES if s.name == args.stage), None)
    if stage is None:
        sys.exit(f"Unknown stage '{args.stage}'. Choose from: {[s.name for s in CURRICULUM_STAGES]}")
    train_stage(
        stage=stage,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_households=args.max_households,
        patience=getattr(args, "patience", 3),
        pretrained_path=args.pretrained,
        tier=args.tier,
        lr=getattr(args, "lr", 5e-4),
        dropout=getattr(args, "dropout", None),
        use_weather=not args.no_weather,
        manifest_path=getattr(args, "manifest_path", None),
        loss=getattr(args, "loss", "smooth_l1"),
        peak_weight=getattr(args, "peak_weight", 4.0),
        peak_quantile=getattr(args, "peak_quantile", 0.95),
        peak_lambda=getattr(args, "peak_lambda", 0.25),
        seed=getattr(args, "seed", 0),
    )


def _cmd_curriculum(args):
    from src.training.curriculum import run_curriculum
    run_curriculum(
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_households=args.max_households,
        tier=args.tier,
        use_weather=not args.no_weather,
        manifest_path=getattr(args, "manifest_path", None),
        seed=getattr(args, "seed", 0),
    )


def _cmd_evaluate(args):
    from src.training.evaluator import evaluate_run
    evaluate_run(
        args.run_dir,
        batch_size=args.batch_size,
        max_households=args.max_households,
        manifest_path=getattr(args, "manifest_path", None),
        split=getattr(args, "split", "test"),
    )


def _cmd_simulate(args):
    from src.simulation import runner
    argv = [
        "simulate",
        "--num_agents", str(args.num_agents),
        "--num_districts", str(args.num_districts),
        "--ticks", str(args.ticks),
        "--pred_len", str(args.pred_len),
    ]
    if args.teacher_run:
        argv += ["--teacher_run", args.teacher_run]
    if args.scaler_path:
        argv += ["--scaler_path", args.scaler_path]
    if getattr(args, "scenario", None):
        argv += ["--scenario", args.scenario]
    if getattr(args, "enable_recommendations", False):
        argv += ["--enable_recommendations"]
    saved = sys.argv
    try:
        sys.argv = argv
        sys.exit(runner.main())
    finally:
        sys.argv = saved


def _cmd_benchmark_summary(args):
    from src.reporting.benchmark_summary import (
        build_benchmark_summary,
        write_benchmark_summary,
    )

    summary = build_benchmark_summary(
        etl_run_dir=getattr(args, "etl_run_dir", None),
        sampling_run_dir=getattr(args, "sampling_run_dir", None),
        progression_report=getattr(args, "progression_report", None),
        evaluation_report=getattr(args, "evaluation_report", None),
        simulation_run_dir=getattr(args, "simulation_run_dir", None),
    )
    out_dir = write_benchmark_summary(summary, output_root=args.output_root)
    print(f"[benchmark-summary] Wrote: {out_dir / 'benchmark_summary.json'}")
    print(f"[benchmark-summary] Wrote: {out_dir / 'benchmark_summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser("neurogrid")
    sub = p.add_subparsers(dest="command", required=True)

    p_etl = sub.add_parser("etl")
    p_etl.add_argument("--max_households", type=int, default=None,
                       help="Alphabetical-LIMIT fallback. Prefer --household_manifest.")
    p_etl.add_argument("--household_manifest", default=None,
                       help="Path to sampling manifest JSON from `sample-households`.")
    p_etl.add_argument("--source_timezone", default=None,
                       help="IANA timezone for naive source timestamps (default HEAPO_TIMEZONE or UTC).")
    p_etl.set_defaults(func=_cmd_etl)

    p_samp = sub.add_parser(
        "sample-households",
        help="Spark MLlib KMeans-stratified representative sampling.",
    )
    p_samp.add_argument("--target_n", type=int, required=True,
                        help="Number of households to keep in the cohort.")
    p_samp.add_argument("--k", type=int, default=5, help="KMeans cluster count.")
    p_samp.add_argument("--seed", type=int, default=0)
    p_samp.add_argument("--data_dir", default=None,
                        help="HEAPO 15-min CSV dir (defaults to env HEAPO_DATA_DIR).")
    p_samp.add_argument("--output_root", default="artifacts/sampling")
    p_samp.set_defaults(func=_cmd_sample_households)

    p_fd = sub.add_parser("forecast-daily",
                          help="Train a 24-hour next-day forecaster (pred_len=96).")
    p_fd.add_argument("--output_dir", default="src/models/stored/forecast_daily")
    p_fd.add_argument("--epochs", type=int, default=10)
    p_fd.add_argument("--batch_size", type=int, default=64)
    p_fd.add_argument("--seq_len", type=int, default=96)
    p_fd.add_argument("--lr", type=float, default=5e-4)
    p_fd.add_argument("--dropout", type=float, default=None,
                      help="Override tier-default dropout.")
    p_fd.add_argument("--loss", default="peak_weighted_smooth_l1",
                      choices=["smooth_l1", "peak_weighted_smooth_l1", "blended_peak_smooth_l1"])
    p_fd.add_argument("--peak_weight", type=float, default=4.0)
    p_fd.add_argument("--peak_quantile", type=float, default=0.95)
    p_fd.add_argument("--peak_lambda", type=float, default=0.25)
    p_fd.add_argument("--seed", type=int, default=0)
    p_fd.add_argument("--patience", type=int, default=3)
    p_fd.add_argument("--max_households", type=int, default=None,
                      help="Optional fallback; prefer --manifest_path for sampled-cohort training.")
    p_fd.add_argument("--pretrained", default=None,
                      help="Optional warm-start checkpoint (e.g. curriculum_h3h/model.pth).")
    p_fd.add_argument("--tier", default="city", choices=["city", "district", "building"])
    p_fd.add_argument("--no_weather", action="store_true")
    p_fd.add_argument("--manifest_path", default=None,
                      help="Sampling manifest JSON for cluster-stratified daily training.")
    p_fd.set_defaults(func=_cmd_forecast_daily)

    p_train = sub.add_parser("train")
    p_train.add_argument("--stage", required=True)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--seq_len", type=int, default=96)
    p_train.add_argument("--max_households", type=int, default=None)
    p_train.add_argument("--pretrained", default=None)
    p_train.add_argument("--tier", default="city", choices=["city", "district", "building"])
    p_train.add_argument("--loss", default="smooth_l1",
                         choices=["smooth_l1", "peak_weighted_smooth_l1", "blended_peak_smooth_l1"])
    p_train.add_argument("--peak_weight", type=float, default=4.0)
    p_train.add_argument("--peak_quantile", type=float, default=0.95)
    p_train.add_argument("--peak_lambda", type=float, default=0.25)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--no_weather", action="store_true")
    p_train.add_argument("--manifest_path", default=None,
                         help="Sampling manifest JSON for cluster-stratified split.")
    p_train.set_defaults(func=_cmd_train)

    p_cur = sub.add_parser("curriculum")
    p_cur.add_argument("--output_root", default="src/models/stored")
    p_cur.add_argument("--epochs", type=int, default=10)
    p_cur.add_argument("--batch_size", type=int, default=64)
    p_cur.add_argument("--seq_len", type=int, default=96)
    p_cur.add_argument("--max_households", type=int, default=None)
    p_cur.add_argument("--tier", default="city", choices=["city", "district", "building"])
    p_cur.add_argument("--no_weather", action="store_true")
    p_cur.add_argument("--manifest_path", default=None,
                       help="Sampling manifest JSON for cluster-stratified split.")
    p_cur.add_argument("--seed", type=int, default=0)
    p_cur.set_defaults(func=_cmd_curriculum)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--run_dir", required=True)
    p_eval.add_argument("--batch_size", type=int, default=64)
    p_eval.add_argument("--max_households", type=int, default=None)
    p_eval.add_argument("--manifest_path", default=None,
                        help="Sampling manifest used at training time (recommended).")
    p_eval.add_argument("--split", default="test", choices=["val", "test"],
                        help="With manifest_path, defaults to held-out test split.")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--num_agents", type=int, default=9)
    p_sim.add_argument("--num_districts", type=int, default=3)
    p_sim.add_argument("--ticks", type=int, default=150)
    p_sim.add_argument("--pred_len", type=int, default=4)
    p_sim.add_argument("--teacher_run", default=None)
    p_sim.add_argument("--scaler_path", default=None)
    p_sim.add_argument("--scenario", default="baseline")
    p_sim.add_argument("--enable_recommendations", action="store_true")
    p_sim.set_defaults(func=_cmd_simulate)

    p_bench = sub.add_parser(
        "benchmark-summary",
        help="Roll up ETL, sampling, training, and simulation artifacts into one report.",
    )
    p_bench.add_argument("--etl_run_dir", default=None,
                         help="Optional explicit ETL run dir (defaults to latest in artifacts/etl_runs).")
    p_bench.add_argument("--sampling_run_dir", default=None,
                         help="Optional explicit sampling run dir (defaults to latest in artifacts/sampling).")
    p_bench.add_argument("--progression_report", default=None,
                         help="Optional explicit progression_report.json path.")
    p_bench.add_argument("--evaluation_report", default=None,
                         help="Optional explicit evaluation_report.json path.")
    p_bench.add_argument("--simulation_run_dir", default=None,
                         help="Optional explicit simulation run dir (defaults to latest in artifacts/simulation_runs).")
    p_bench.add_argument("--output_root", default="artifacts/benchmarks")
    p_bench.set_defaults(func=_cmd_benchmark_summary)

    # ---- etl-spark-fl ---------------------------------------------------- #
    p_etl_sfl = sub.add_parser(
        "etl-spark-fl",
        help="Spark ETL into the dedicated Spark-FL Parquet root.",
    )
    p_etl_sfl.add_argument("--output_dir", default=None,
                           help="Override output. Defaults to data/processed_parquet_spark_fl.")
    p_etl_sfl.add_argument("--max_households", type=int, default=None)
    p_etl_sfl.add_argument("--source_timezone", default=None)
    p_etl_sfl.add_argument("--household_manifest", default=None)
    p_etl_sfl.set_defaults(func=_cmd_etl_spark_fl)

    # ---- train-spark-ddp ------------------------------------------------- #
    p_ddp = sub.add_parser(
        "train-spark-ddp",
        help="TorchDistributor + DDP city-teacher pretraining (centralized big-data).",
    )
    p_ddp.add_argument("--parquet_root", default="data/processed_parquet_spark_fl")
    p_ddp.add_argument("--output_dir", default="src/models/stored/spark_ddp_teacher")
    p_ddp.add_argument("--tier", default="city", choices=["city", "district", "building"])
    p_ddp.add_argument("--pred_len", type=int, default=4)
    p_ddp.add_argument("--seq_len", type=int, default=24)
    p_ddp.add_argument("--epochs", type=int, default=5)
    p_ddp.add_argument("--batch_size", type=int, default=32)
    p_ddp.add_argument("--lr", type=float, default=5e-4)
    p_ddp.add_argument("--world_size", type=int, default=2,
                       help="Number of DDP ranks (TorchDistributor processes).")
    p_ddp.add_argument("--seed", type=int, default=0)
    p_ddp.add_argument("--manifest_path", default=None,
                       help="Spark MLlib KMeans sampling manifest (recommended).")
    p_ddp.set_defaults(func=_cmd_train_spark_ddp)

    # ---- simulate-spark-fl ----------------------------------------------- #
    p_sfl = sub.add_parser(
        "simulate-spark-fl",
        help="Spark + TorchDistributor federated-learning rounds (rank-isolated).",
    )
    p_sfl.add_argument("--num_agents", type=int, default=8)
    p_sfl.add_argument("--num_districts", type=int, default=2)
    p_sfl.add_argument("--rounds", type=int, default=5)
    p_sfl.add_argument("--f_byzantine", type=int, default=1,
                       help="Per-district Byzantine tolerance for Multi-Krum.")
    p_sfl.add_argument("--byzantine_fraction", type=float, default=0.0,
                       help="Fraction of ranks to designate as attackers.")
    p_sfl.add_argument("--byzantine_attack", default="scale",
                       choices=["scale", "flip", "noise"])
    p_sfl.add_argument("--clip_threshold", type=float, default=1.0)
    p_sfl.add_argument("--trim_ratio", type=float, default=0.1)
    p_sfl.add_argument("--convergence_epsilon", type=float, default=1e-3)
    p_sfl.add_argument("--convergence_patience", type=int, default=2)
    p_sfl.add_argument("--spark_master", default="local[*]")
    p_sfl.add_argument("--resume_from_round", type=int, default=None,
                       help="Skip rounds < this and load W from prior run.")
    p_sfl.add_argument("--parquet_root", default="data/processed_parquet_spark_fl")
    p_sfl.add_argument("--run_root", default="artifacts/spark_fl/runs")
    # Building-local SGD budget.
    p_sfl.add_argument("--local_steps_building", type=int, default=5)
    # Local SGD knobs.
    p_sfl.add_argument("--pred_len", type=int, default=4)
    p_sfl.add_argument("--seq_len", type=int, default=24)
    p_sfl.add_argument("--tier", default="building",
                       choices=["city", "district", "building"])
    p_sfl.add_argument("--lr", type=float, default=1e-3)
    p_sfl.add_argument("--batch_size", type=int, default=8)
    p_sfl.add_argument("--dp_sigma", type=float, default=0.0)
    p_sfl.add_argument("--dp_clip_C", type=float, default=1.0)
    p_sfl.add_argument("--topk_ratio", type=float, default=0.1)
    p_sfl.add_argument("--seed", type=int, default=0)
    p_sfl.add_argument("--manifest_path", default=None,
                       help="Spark MLlib KMeans sampling manifest (strongly "
                            "recommended). Provides cluster-stratified, "
                            "generalization-friendly cohort + cluster-balanced "
                            "district assignment.")
    p_sfl.set_defaults(func=_cmd_simulate_spark_fl)

    # ---- benchmark-scalability ------------------------------------------- #
    p_bsc = sub.add_parser(
        "benchmark-scalability",
        help="Sweep agent counts; compare gRPC vs Spark FL throughput.",
    )
    p_bsc.add_argument("--agent_counts", type=int, nargs="+", default=[16, 64, 256])
    p_bsc.add_argument("--rounds", type=int, default=2)
    p_bsc.add_argument("--num_districts", type=int, default=4)
    p_bsc.add_argument("--local_steps_building", type=int, default=3)
    p_bsc.add_argument("--parquet_root", default="data/processed_parquet_spark_fl")
    p_bsc.add_argument("--output_root", default="artifacts/spark_fl/benchmarks")
    p_bsc.add_argument("--include_grpc", action="store_true",
                       help="Also run the gRPC path for direct comparison.")
    p_bsc.set_defaults(func=_cmd_benchmark_scalability)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
