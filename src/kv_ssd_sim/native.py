"""Version-pinned native vLLM construction and a CPU event adapter.

The Scheduler, KVCacheManager, MooncakeStoreScheduler, Store layout planning,
transfer threads, and MooncakeDistributedStore are upstream implementations.
The worker's CUDA save fence is replaced with a host event, and explicit
offload parameters are forwarded to native Store setup. No model is
constructed and no forward pass is called.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from .events import EventLog
from .payload import PreparedPayloads
from .runtime import PROJECT_ROOT, free_port
from .storage import ObservedStore

VLLM_VERSION = "0.34.0rc1.dev302+gbe47189f3.cpu"


class HostFence:
    def __init__(self):
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def release(self) -> None:
        self._ready.set()

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._ready.set()

    def synchronize(self) -> None:
        if not self._ready.wait(timeout=30):
            raise TimeoutError("Simulated compute did not release the save fence")
        if self._error is not None:
            raise RuntimeError("Simulated compute failed") from self._error


class CPUWorkerAdapter:
    """Delegate native worker behavior, replacing only CUDA save-event creation."""

    def __init__(self, native):
        self.native = native
        self.fence: HostFence | None = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def wait_for_save(self, metadata):
        if self.native._capacity_only or not self.native.can_put:
            return
        if self.fence is None:
            raise RuntimeError("A host compute fence is required for each step")
        for request in metadata.requests:
            if not request.can_save:
                continue
            request.current_event = self.fence
            if self.native.kv_send_thread is None:
                raise RuntimeError("Native KV sending thread is not initialized")
            self.native.kv_send_thread.add_request(request)


class NativeRuntime:
    def __init__(
        self,
        output: Path,
        events: EventLog,
        *,
        read_delay_ms: float,
        max_model_len: int = 8192,
        num_blocks: int = 128,
        hybrid_config=None,
    ):
        installed = importlib.metadata.version("vllm")
        if installed != VLLM_VERSION:
            raise RuntimeError(f"This adapter requires vllm=={VLLM_VERSION}, found {installed}")
        import torch
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            KVTransferConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
            worker as worker_module,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.connector import (
            MooncakeStoreConnector,
        )
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.structured_output import StructuredOutputManager

        torch.set_num_threads(1)
        self.events = events
        self._stack = ExitStack()
        self.torch = torch
        self.block_size = 64
        self.num_blocks = num_blocks
        self.hybrid = None
        if hybrid_config is not None:
            from .hybrid import HybridBuffers

            self.hybrid = HybridBuffers(hybrid_config, num_blocks)
        self.layer_names = (
            self.hybrid.names
            if self.hybrid
            else [f"model.layers.{i}.self_attn.attn" for i in range(2)]
        )
        model_path = PROJECT_ROOT / "configs/model"
        self.config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path),
                skip_tokenizer_init=True,
                dtype="float16",
                max_model_len=max_model_len,
                hf_overrides={"max_position_embeddings": max_model_len},
                seed=0,
                enforce_eager=True,
            ),
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                is_encoder_decoder=False,
                max_num_seqs=8,
                max_num_batched_tokens=128,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
                async_scheduling=False,
                watermark=0.0,
            ),
            cache_config=CacheConfig(
                block_size=self.block_size,
                enable_prefix_caching=True,
                cache_dtype="auto",
                gpu_memory_utilization=0.5,
                mamba_cache_mode="align" if self.hybrid else "none",
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "cache_prefix": (
                        "kv-ssd-sim-hybrid-" + hybrid_config.fingerprint
                        if self.hybrid
                        else "kv-ssd-sim-v1"
                    ),
                    **({"store_tp_size": 1} if self.hybrid else {}),
                    "load_async": True,
                    "lookup_async": False,
                    "save_decode_cache": False,
                },
            ),
        )
        self._stack.enter_context(set_current_vllm_config(self.config))
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"tcp://127.0.0.1:{free_port()}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo"
        )
        spec = FullAttentionSpec(
            block_size=self.block_size, num_kv_heads=2, head_size=32, dtype=torch.float16
        )
        self.kv_cache_config = KVCacheConfig(
            num_blocks=self.num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=(
                self.hybrid.groups if self.hybrid else [KVCacheGroupSpec(self.layer_names, spec)]
            ),
        )
        if self.hybrid:
            self.config.cache_config.kv_cache_layout = "LBHNC"
            self.config.cache_config.prefix_match_unit = 64
            self.config.cache_config.hash_block_size = 64
            self.config.cache_config.cache_hit_alignment_tokens = 64
        self.config.cache_config.num_gpu_blocks = self.num_blocks
        register_all_kvcache_specs(self.config)

        # This pinned upstream setup helper does not forward enable_offload or
        # offload_dir to the native Store. Scope the compatibility adapter to
        # construction; every transfer/layout/lookup still uses upstream code.
        def setup_cpu_offload(store, config, hostname, engine=None):
            if engine is not None or config.mode != "embedded" or config.tenant_id != "default":
                raise ValueError("CPU prototype requires a private embedded default-tenant Store")
            code = store.setup(
                hostname,
                config.metadata_server,
                config.global_segment_size,
                config.local_buffer_size,
                config.protocol,
                config.device_name,
                config.master_server_address,
                None,
                True,
                str((output / "disk").resolve()),
            )
            if code:
                raise RuntimeError(f"Native Mooncake offload setup failed: {code}")

        with patch.object(worker_module, "setup_mooncake_store", setup_cpu_offload):
            self.worker_connector = MooncakeStoreConnector(
                self.config,
                KVConnectorRole.WORKER,
                self.kv_cache_config,
            )
        native_worker = self.worker_connector.connector_worker
        self.store = ObservedStore(native_worker.store, events, read_delay_ms=read_delay_ms)
        native_worker.store = self.store
        self.worker = CPUWorkerAdapter(native_worker)
        self.worker_connector.connector_worker = self.worker
        self.caches = {
            name: torch.zeros((self.num_blocks, 2, self.block_size, 2, 32), dtype=torch.float16)
            for name in self.layer_names
        }
        if self.hybrid:
            self.caches = self.hybrid.caches
        self.worker_connector.register_kv_caches(self.caches)
        if self.hybrid and native_worker.store_tp_size != 1:
            raise RuntimeError("Native hybrid Store layout unexpectedly fell back")
        for thread in native_worker.kv_recv_threads:
            thread._handle_request = self.store.trace_request_handler(thread._handle_request)
        if native_worker.kv_send_thread is not None:
            thread = native_worker.kv_send_thread
            thread._handle_request = self.store.trace_request_handler(thread._handle_request)
        self.scheduler = Scheduler(
            vllm_config=self.config,
            kv_cache_config=self.kv_cache_config,
            block_size=self.block_size,
            log_stats=False,
            structured_output_manager=StructuredOutputManager(self.config),
        )
        if self.hybrid:
            events.emit("hybrid_initialized", **self.hybrid.describe())
        self.payloads = None
        self.cache_byte_views = (
            []
            if self.hybrid
            else [
                tensor.numpy().view("uint8").reshape(self.num_blocks, 2, self.block_size, 2, 2, 32)
                for tensor in self.caches.values()
            ]
        )
        source_path = Path(inspect.getfile(Scheduler))
        events.emit(
            "runtime_initialized",
            scheduler_class=f"{Scheduler.__module__}.{Scheduler.__name__}",
            scheduler_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            vllm_version=installed,
            cache_bytes=sum(
                t.numel() * t.element_size()
                for t in (self.hybrid.pools if self.hybrid else self.caches.values())
            ),
            num_blocks=self.num_blocks,
            block_size=self.block_size,
        )

    def prepare_payloads(self, requests) -> dict:
        started = time.perf_counter()
        self.payloads = PreparedPayloads(requests, num_layers=len(self.caches))
        report = {
            "preparation_ms": (time.perf_counter() - started) * 1000,
            "prepared_bytes": self.payloads.nbytes,
            "placement": "before replay clock and request admission",
        }
        self.events.emit("payloads_prepared", **report)
        return report

    def materialize(self, request, start: int, end: int) -> None:
        """Copy only scheduled slots; no hashing or tensor creation on this path."""
        if self.payloads is None:
            raise RuntimeError("Prepare placeholder payloads before starting replay")
        if self.hybrid:
            self.hybrid.materialize(self, request, start, end)
            return
        blocks = self.scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()[0]
        self.payloads.copy_range(
            request.request_id, start, end, blocks, self.cache_byte_views, self.block_size
        )

    def close(self) -> None:
        try:
            self.worker_connector.shutdown()
        finally:
            from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
            self._stack.close()
