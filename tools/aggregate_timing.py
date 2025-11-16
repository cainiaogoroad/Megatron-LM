import os
import sys
import json
import argparse
import glob
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate timing JSONL records written by MegatronCollector"
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory containing timing files. Default: {VTIMELINE_LOGGER_DIR}/Collector or current dir",
    )
    parser.add_argument(
        "--pattern",
        default="timing_*.jsonl",
        help="Glob pattern to match timing files (default: timing_*.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path (.json or .csv). If omitted, prints to stdout",
    )
    parser.add_argument(
        "--group-by",
        default="stage+op",
        choices=["stage", "op", "stage+op", "stage+op+name"],
        help="Grouping key for aggregation (default: stage+op)",
    )
    parser.add_argument(
        "--metrics",
        default="t_copy_ms,t_hash_ms,grad_t_copy_ms,grad_t_hash_ms,t_json_ms,t_insert_ms,bytes,size_bytes",
        help="Comma-separated metrics to aggregate (only metrics present in records are used)",
    )
    return parser.parse_args()


def find_files(base_dir: str, pattern: str):
    return sorted(glob.glob(os.path.join(base_dir, pattern)))


def read_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def group_key(rec: dict, mode: str):
    stage = rec.get("stage")
    op = rec.get("op")
    name = rec.get("name")
    if mode == "stage":
        return (stage,)
    if mode == "op":
        return (op,)
    if mode == "stage+op":
        return (stage, op)
    if mode == "stage+op+name":
        return (stage, op, name)
    return (stage, op)


def aggregate(files, metrics, mode):
    # maps key -> { metric: {sum: float, count: int} }, plus record_count
    agg = defaultdict(lambda: {"_records": 0})
    for fp in files:
        for rec in read_jsonl(fp):
            k = group_key(rec, mode)
            bucket = agg[k]
            bucket["_records"] += 1
            for m in metrics:
                if m in rec and rec[m] is not None:
                    mt = bucket.get(m)
                    if mt is None:
                        bucket[m] = {"sum": 0.0, "count": 0}
                        mt = bucket[m]
                    try:
                        mt["sum"] += float(rec[m])
                        mt["count"] += 1
                    except Exception:
                        # Non-numeric value, skip
                        pass
    # finalize
    results = []
    for k, bucket in agg.items():
        entry = {
            "group": {
                "stage": k[0] if len(k) > 0 else None,
                "op": k[1] if len(k) > 1 else None,
                "name": k[2] if len(k) > 2 else None,
            },
            "records": bucket["_records"],
            "metrics": {},
        }
        for m in metrics:
            mt = bucket.get(m)
            if mt and mt["count"] > 0:
                entry["metrics"][m] = {
                    "avg": mt["sum"] / mt["count"],
                    "count": mt["count"],
                    "sum": mt["sum"],
                }
        results.append(entry)
    # sort by stage/op then name for readability
    results.sort(key=lambda e: (
        str(e["group"]["stage"]), str(e["group"]["op"]), str(e["group"]["name"]))
    )
    return {
        "total_groups": len(results),
        "total_records": sum(e["records"] for e in results),
        "group_by": mode,
        "results": results,
    }


def write_output(data, out_path: str | None):
    if not out_path:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".json":
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif ext == ".csv":
        # flatten to CSV rows: stage,op,name,metric,avg,count,sum
        import csv

        rows = []
        for g in data["results"]:
            stage = g["group"]["stage"]
            op = g["group"]["op"]
            name = g["group"].get("name")
            for m, v in g["metrics"].items():
                rows.append({
                    "stage": stage,
                    "op": op,
                    "name": name,
                    "metric": m,
                    "avg": v["avg"],
                    "count": v["count"],
                    "sum": v["sum"],
                })
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "op", "name", "metric", "avg", "count", "sum"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        # default to JSON
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    base_dir = args.dir
    if base_dir is None:
        vt_dir = os.environ.get("VTIMELINE_LOGGER_DIR")
        if vt_dir:
            base_dir = os.path.join(vt_dir, "Collector")
        else:
            base_dir = os.getcwd()

    files = find_files(base_dir, args.pattern)
    if not files:
        print(f"No timing files found in {base_dir} with pattern {args.pattern}", file=sys.stderr)
        sys.exit(1)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    data = aggregate(files, metrics, args.group_by)
    write_output(data, args.out)


if __name__ == "__main__":
    main()