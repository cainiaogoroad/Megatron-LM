#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib
from typing import Dict, Any, Optional

# NOTE: You might need to import Megatron-LM specific utilities to get some of the required data.
# from megatron.core import mpu
# from megatron.utils import get_args

def _get_cksum(data: torch.Tensor):
    """Computes the SHA256 checksum of a torch.Tensor."""
    if not isinstance(data, torch.Tensor):
        # Return None if data is not a tensor, e.g., for empty grads
        return None
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)
    return hasher.hexdigest()


class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))
    
    # Class attributes to be initialized
    db_: Optional[duckdb.DuckDBPyConnection] = None
    model_ = None
    optimizer_ = None
    scheduler_ = None
    ranks_info_: Dict[str, Any] = {}

    def __init__(self):
        raise RuntimeError("Use MegatronCollector.initialize() to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        """Initializes the collector, database connection, and table schema."""
        if cls.db_ is not None:
            return

        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_") and cls.ranks_info_, "The rank information must be set via set_process_group_info"

        # Build a more descriptive filename using all available rank info
        rank_keys = sorted(cls.ranks_info_.keys())
        filename_parts = [f"{key}{cls.ranks_info_[key]}" for key in rank_keys if 'rank' in key or 'size' in key]
        db_filename = "coredump_{}.db".format("_".join(filename_parts))
        db_path = os.path.join(db_dir, db_filename)

        cls.db_ = duckdb.connect(db_path)
        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def set_process_group_info(cls, ranks_info: Dict[str, Any]):
        """Sets the process group information for the current rank."""
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        """Sets the core components to be tracked."""
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        if cls.db_ is None:
            cls.initialize()

    @classmethod
    def _insert_data(cls, stage_name: str, data: Dict[str, Any]):
        """A centralized method to insert data into the database."""
        if not cls.should_dump():
            return
        
        # Automatically enrich data with rank information
        enriched_data = data.copy()
        enriched_data.update(cls.ranks_info_)
        
        try:
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, json.dumps(enriched_data)),
            )
        except Exception as e:
            print(f"Error inserting data into coredump for stage '{stage_name}': {e}")

    @classmethod
    def should_dump(cls):
        """Determines if data should be dumped for the current step."""
        return cls.step_ <= cls.dump_step_

    @classmethod
    def dump_main_grad(cls, param, param_name: str, stage_name: str = "backward"):
        """Dumps the main_grad of a specific parameter."""
        param_info = {
            "type": "main_grad",
            "name": param_name,
            "cksum": _get_cksum(param.main_grad),
            "shape": list(param.main_grad.shape) if param.main_grad is not None else None,
            "dtype": str(param.main_grad.dtype) if param.main_grad is not None else None,
        }
        cls._insert_data(stage_name, param_info)

    @classmethod
    def dump_main_param(cls, stage_name: str):
        """Dumps the main_param for all model parameters."""
        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = hasattr(param, "main_param") and param.main_param is not None
                param_info = {
                    "type": "main_param",
                    "name": name,
                    "cksum": _get_cksum(param.main_param) if main_param_exist else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "dtype": str(param.main_param.dtype) if main_param_exist else None,
                }
                cls._insert_data(stage_name, param_info)

    @classmethod
    def dump_model(cls, stage_name: str):
        """Dumps model parameters and their gradients."""
        for model in cls.model_:
            for name, param in model.named_parameters():
                param_info = {
                    "type": "model_param",
                    "name": name,
                    "cksum": _get_cksum(param),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": _get_cksum(param.grad) if param.grad is not None else None,
                    "grad_shape": list(param.grad.shape) if param.grad is not None else None,
                    "grad_dtype": str(param.grad.dtype) if param.grad is not None else None,
                }
                cls._insert_data(stage_name, param_info)
    
    # =========================================================================
    # NEWLY ADDED METHODS BASED ON YOUR REQUIREMENTS
    # =========================================================================

    @classmethod
    def dump_optimizer_state(cls, stage_name: str):
        """
        Dumps detailed optimizer state for each parameter group and parameter.
        Suggested stages: 'before-optimizer-step', 'after-optimizer-step'
        """
        optimizer = cls.optimizer_
        if optimizer is None:
            return

        # 1. Dump AMP/FP16 loss scaler state
        # NOTE: You need to get the loss scaler object from your training script.
        # It could be part of the optimizer (e.g., in Apex) or a separate GradScaler object.
        loss_scaler = getattr(optimizer, 'loss_scaler', None)
        if loss_scaler:
            scaler_info = {
                "type": "loss_scaler",
                "loss_scale": loss_scaler.get_scale(),
                "is_dynamic": hasattr(loss_scaler, 'is_dynamic') and loss_scaler.is_dynamic,
            }
            cls._insert_data(stage_name, scaler_info)
        
        # 2. Dump param_groups and parameter-specific states
        for i, group in enumerate(optimizer.param_groups):
            group_info = {
                "type": "optimizer_group",
                "group_index": i,
                "lr": group['lr'],
                "betas": list(group.get('betas', [])),
                "eps": group.get('eps'),
                "weight_decay": group.get('weight_decay'),
                "captured_step": group.get('captured_step', None), # For FSDP
            }
            cls._insert_data(stage_name, group_info)
            
            for param in group['params']:
                if param not in optimizer.state:
                    continue
                
                state = optimizer.state[param]
                
                # ZeRO/Distributed Optimizer Info
                # NOTE: These attributes are specific to DeepSpeed's ZeRO or other distributed optimizers.
                # You may need to adjust the attribute names.
                is_offloaded = hasattr(param, 'ds_tensor') and param.ds_tensor.status == 'NOT_AVAILABLE'
                owner_rank = getattr(param, 'ds_id', None) # Example for DeepSpeed
                
                state_info = {
                    "type": "optimizer_param_state",
                    "param_name": [name for name, p in cls.model_[0].named_parameters() if p is param][0],
                    "step": state.get('step'),
                    "exp_avg_cksum": _get_cksum(state.get('exp_avg')),
                    "exp_avg_sq_cksum": _get_cksum(state.get('exp_avg_sq')),
                    "has_master_fp32": hasattr(param, 'main_param'),
                    "master_fp32_dtype": str(param.main_param.dtype) if hasattr(param, 'main_param') and param.main_param is not None else None,
                    "master_fp32_device": str(param.main_param.device) if hasattr(param, 'main_param') and param.main_param is not None else None,
                    "zero_owner_rank": owner_rank,
                    "zero_is_offloaded_to_cpu": is_offloaded,
                }
                cls._insert_data(stage_name, state_info)

    @classmethod
    def dump_scheduler_state(cls, stage_name: str = "after-scheduler-step"):
        """Dumps the state of the learning rate scheduler."""
        scheduler = cls.scheduler_
        if scheduler is None:
            return
            
        # NOTE: Attributes are scheduler-dependent. Add what's important for your scheduler.
        scheduler_info = {
            "type": "scheduler_state",
            "last_epoch": getattr(scheduler, 'last_epoch', None),
            "warmup_steps": getattr(scheduler, 'warmup_steps', None),
            "num_steps": getattr(scheduler, 'num_steps', None),
            "base_lrs": getattr(scheduler, 'base_lrs', None),
        }
        cls._insert_data(stage_name, scheduler_info)

    @classmethod
    def dump_parallel_metadata(cls, stage_name: str, extra_meta: Dict[str, Any]):
        """
        Dumps parallel process group info and other scheduling metadata.
        This should be called with context from the training loop.
        
        Example `extra_meta`:
        {
            "microbatch_id": current_microbatch_id,
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "virtual_pipeline_model_parallel_rank": mpu.get_virtual_pipeline_model_parallel_rank(),
        }
        """
        meta_info = {
            "type": "parallel_metadata",
            **extra_meta
        }
        cls._insert_data(stage_name, meta_info)
        
    @classmethod
    def dump_rng_state(cls, stage_name: str):
        """
        Dumps the checksum of CPU and CUDA RNG states.
        Suggested stages: 'pre-forward', 'pre-microbatch'
        """
        # NOTE: Megatron's parallel RNG states need to be accessed via their tracker.
        # from megatron.core.parallel_state import get_data_parallel_rng_tracker
        # tp_rng_tracker = get_data_parallel_rng_tracker() ...
        
        rng_info = {
            "type": "rng_state",
            "cpu_rng_state_cksum": _get_cksum(torch.get_rng_state()),
            "cuda_rng_state_cksum": _get_cksum(torch.cuda.get_rng_state()),
            # "megatron_tp_rng_cksum": _get_cksum(tp_rng_tracker.get_states()) # Example
        }
        cls._insert_data(stage_name, rng_info)

    @classmethod
    def dump_buffers(cls, stage_name: str):
        """Dumps model buffers (e.g., RoPE tables, biases)."""
        for model in cls.model_:
            for name, buf in model.named_buffers():
                buffer_info = {
                    "type": "model_buffer",
                    "name": name,
                    "cksum": _get_cksum(buf),
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype),
                }
                cls._insert_data(stage_name, buffer_info)

    @classmethod
    def dump_fp8_state(cls, stage_name: str = "after-forward-per-mb"):
        """Dumps FP8-related metadata like amax history and scales."""
        # NOTE: This assumes FP8 state is stored in a central place or is accessible.
        # This is a conceptual implementation. You'll need to adapt it to your FP8 framework.
        # from megatron.core.tensor_parallel.layers import Fp8Linear
        
        for model in cls.model_:
            for name, module in model.named_modules():
                # if isinstance(module, Fp8Linear): # Example check
                if hasattr(module, "fp8_meta"): # A more generic check
                    fp8_meta = module.fp8_meta
                    meta_info = {
                        "type": "fp8_metadata",
                        "module_name": name,
                        "amax_history_cksum": _get_cksum(fp8_meta.get('amax_history')),
                        "scale_cksum": _get_cksum(fp8_meta.get('scale')),
                        "scale_inv_cksum": _get_cksum(fp8_meta.get('scale_inv')),
                        "fp8_enabled": True, # Or some other flag from your config
                        "fp8_recipe": "E4M3", # Example, get from args
                    }
                    cls._insert_data(stage_name, meta_info)

    @classmethod
    def dump_moe_state(cls, stage_name: str):
        """Dumps MoE router statistics and metadata."""
        # NOTE: This is a conceptual implementation. You'll need to adapt it to your MoE layer implementation.
        # from megatron.core.transformer.moe.moe_layer import MoELayer
        
        for model in cls.model_:
            for name, module in model.named_modules():
                # if isinstance(module, MoELayer): # Example check
                # A generic check for router stats, assuming the module saves them after forward.
                if hasattr(module, "router_stats"):
                    stats = module.router_stats 
                    moe_info = {
                        "type": "moe_metadata",
                        "module_name": name,
                        "tokens_per_expert": stats.get("tokens_per_expert_hist", []).tolist(),
                        "dropped_tokens": stats.get("dropped_tokens_count"),
                        "capacity_factor": stats.get("capacity_factor"),
                        "grouped_gemm_enabled": stats.get("grouped_gemm"),
                    }
                    cls._insert_data(stage_name, moe_info)

    @classmethod
    def step(cls):
        """Increments the global step counter."""
        cls.step_ += 1