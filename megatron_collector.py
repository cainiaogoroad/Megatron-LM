#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib
import time


def _get_cksum_and_timing(data: torch.Tensor):
    t0 = time.perf_counter()
    byte_data = (
        data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    )
    t1 = time.perf_counter()
    hasher = hashlib.sha256()
    hasher.update(byte_data)
    t2 = time.perf_counter()

    return hasher.hexdigest(), (t1 - t0), (t2 - t1), len(byte_data)


def _ms(seconds: float) -> int:
    return int(seconds * 1000)


class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "/var/log")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_"), "the rank information must be set"

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_dp{}_tp{}_pp{}_cp{}.db".format(
                cls.ranks_info_["dp"], cls.ranks_info_["tp"],cls.ranks_info_["pp"],cls.ranks_info_["cp"]
            ),
        )
        cls.db_ = duckdb.connect(db_path)

        # Prepare timing file (per-process) to avoid contention
        timing_path = os.path.join(
            root_dir,
            "Collector",
            "timing_dp{}_tp{}_pp{}_cp{}_pid{}.jsonl".format(
                cls.ranks_info_["dp"],
                cls.ranks_info_["tp"],
                cls.ranks_info_["pp"],
                cls.ranks_info_["cp"],
                os.getpid(),
            ),
        )
        try:
            cls.timing_fp_ = open(timing_path, "a", buffering=1)
        except Exception:
            cls.timing_fp_ = None

        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def _write_timing(cls, record: dict):
        if getattr(cls, "timing_fp_", None) is None:
            return
        try:
            cls.timing_fp_.write(json.dumps(record) + "\n")
        except Exception:
            pass

    @classmethod
    def set_process_group_info(cls, ranks_info):
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls.initialize()

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "backward"
    ):
        if not cls.should_dump():
            return

        cksum, t_copy, t_hash, nbytes = _get_cksum_and_timing(param.main_grad)
        param_info = {
            "name": param_name,
            "cksum": cksum,
            "shape": list(param.main_grad.shape),
            "type": str(param.main_grad.type()),
        }
        param_info.update(cls.ranks_info_)

        try:
            t0 = time.perf_counter()
            payload = json.dumps(param_info)
            t1 = time.perf_counter()
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, payload),
            )
            t2 = time.perf_counter()
            cls._write_timing(
                {
                    "step": cls.step_,
                    "stage": stage_name,
                    "op": "main_grad",
                    "name": param_name,
                    "t_copy_ms": _ms(t_copy),
                    "t_hash_ms": _ms(t_hash),
                    "t_json_ms": _ms(t1 - t0),
                    "t_insert_ms": _ms(t2 - t1),
                    "size_bytes": len(payload),
                    "pid": os.getpid(),
                }
            )
        except Exception as e:
            print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = (
                    hasattr(param, "main_param") and param.main_param is not None
                )
                cksum = None
                t_copy = None
                t_hash = None
                nbytes = None
                if main_param_exist:
                    cksum, t_copy, t_hash, nbytes = _get_cksum_and_timing(
                        param.main_param
                    )
                param_info = {
                    "name": name,
                    "cksum": cksum if main_param_exist else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "type": str(param.main_param.type())
                    if hasattr(param, "main_param") and param.main_param is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    t0 = time.perf_counter()
                    payload = json.dumps(param_info)
                    t1 = time.perf_counter()
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, payload),
                    )
                    t2 = time.perf_counter()
                    cls._write_timing(
                        {
                            "step": cls.step_,
                            "stage": stage_name,
                            "op": "main_param",
                            "name": name,
                            "t_copy_ms": _ms(t_copy) if t_copy is not None else None,
                            "t_hash_ms": _ms(t_hash) if t_hash is not None else None,
                            "t_json_ms": _ms(t1 - t0),
                            "t_insert_ms": _ms(t2 - t1),
                            "size_bytes": len(payload),
                            "pid": os.getpid(),
                        }
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                p_cksum, p_t_copy, p_t_hash, p_nbytes = _get_cksum_and_timing(param)
                g_cksum = None
                g_t_copy = None
                g_t_hash = None
                g_nbytes = None
                if param.grad is not None:
                    g_cksum, g_t_copy, g_t_hash, g_nbytes = _get_cksum_and_timing(param.grad)
                param_info = {
                    "name": name,
                    "cksum": p_cksum,
                    "shape": list(param.shape),
                    "type": str(param.type()),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": g_cksum if param.grad is not None else None,
                    "grad_shape": list(param.grad.shape)
                    if param.grad is not None
                    else None,
                    "grad_type": str(param.grad.type())
                    if param.grad is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    t0 = time.perf_counter()
                    payload = json.dumps(param_info)
                    t1 = time.perf_counter()
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, payload),
                    )
                    t2 = time.perf_counter()
                    cls._write_timing(
                        {
                            "step": cls.step_,
                            "stage": stage_name,
                            "op": "model_param",
                            "name": name,
                            "t_copy_ms": _ms(p_t_copy),
                            "t_hash_ms": _ms(p_t_hash),
                            "grad_t_copy_ms": _ms(g_t_copy) if g_t_copy is not None else None,
                            "grad_t_hash_ms": _ms(g_t_hash) if g_t_hash is not None else None,
                            "t_json_ms": _ms(t1 - t0),
                            "t_insert_ms": _ms(t2 - t1),
                            "size_bytes": len(payload),
                            "pid": os.getpid(),
                        }
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")


    @classmethod
    def dump_training_batch(cls, tokens, labels, loss_mask, attention_mask, position_ids, stage_name: str = "after-get-batch"):
        """Dump key training batch info after get_batch.

        Records shapes, dtypes, devices, simple stats and checksums to help verify data pipeline.
        """
        if not cls.should_dump():
            return

        def _tinfo(name, t):
            if t is None:
                return None
            try:
                cksum, t_copy, t_hash, nbytes = _get_cksum_and_timing(t)
                cls._write_timing(
                    {
                        "step": cls.step_,
                        "stage": stage_name,
                        "op": "batch_tensor",
                        "name": name,
                        "t_copy_ms": _ms(t_copy),
                        "t_hash_ms": _ms(t_hash),
                        "bytes": int(nbytes),
                        "pid": os.getpid(),
                    }
                )
                return {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                    "cksum": cksum,
                }
            except Exception:
                return None

        try:
            batch_info = {
                "type": "batch",
                "tokens": _tinfo("tokens", tokens),
                "labels": _tinfo("labels", labels),
                "loss_mask": (lambda info: {**info, "sum": float(loss_mask.sum().item())} if info is not None else None)(_tinfo("loss_mask", loss_mask)),
                "attention_mask": (lambda info: {**info, "sum": float(attention_mask.sum().item())} if info is not None else None)(_tinfo("attention_mask", attention_mask)),
                "position_ids": (lambda info: {**info, "min": int(position_ids.min().item()), "max": int(position_ids.max().item())} if info is not None and position_ids.numel() > 0 else info)(_tinfo("position_ids", position_ids)),
            }
            batch_info.update(cls.ranks_info_)
            t0 = time.perf_counter()
            payload = json.dumps(batch_info)
            t1 = time.perf_counter()
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, payload),
            )
            t2 = time.perf_counter()
            cls._write_timing(
                {
                    "step": cls.step_,
                    "stage": stage_name,
                    "op": "batch",
                    "t_json_ms": _ms(t1 - t0),
                    "t_insert_ms": _ms(t2 - t1),
                    "size_bytes": len(payload),
                    "pid": os.getpid(),
                }
            )
        except Exception as e:
            print(f"Error inserting batch data into coredump: {e}")

    @classmethod
    def step(cls):
        cls.step_ += 1
