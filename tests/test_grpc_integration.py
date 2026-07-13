from __future__ import annotations

from concurrent import futures

import grpc
import torch

from src.ismcc.communication import CommunicationModule
from src.proto import neurogrid_pb2_grpc
from src.tiers.city import CityAggregator


def test_real_grpc_roundtrip_applies_delta_and_polls_checkpoint(tmp_path):
    city = CityAggregator(
        expected_districts=1,
        audit_dir=tmp_path / "audit",
        state_dir=tmp_path / "state",
        max_relative_update=0.5,
    )
    city.global_state = {"w": torch.tensor([10.0])}
    city.store.save_baseline(city.global_state)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    neurogrid_pb2_grpc.add_FederatedAggregatorServicer_to_server(city, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    client = CommunicationModule(
        "D1",
        district_host="127.0.0.1",
        district_port=str(port),
        topk_ratio=1.0,
        model_version="city-round-0",
    )
    try:
        assert client.upload_delta({"w": torch.tensor([1.0])}, n_samples=4)
        checkpoint = client.pull_global()
        assert checkpoint is not None
        assert torch.equal(checkpoint["w"], torch.tensor([11.0]))
        assert client.current_round == 1
    finally:
        client.close()
        server.stop(grace=0).wait()
