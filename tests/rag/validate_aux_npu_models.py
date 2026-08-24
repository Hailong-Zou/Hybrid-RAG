#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path


def read_text(path):
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def parse_ini(path):
    sections = {}
    current = None
    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = {}
        elif current and "=" in line:
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def runtime_evidence(server_log, model_path):
    match = re.search(
        r"\[(\d+)\].*load_model: loading model '" + re.escape(model_path) + r"'",
        server_log,
    )
    if not match:
        return {"pid": None, "htp0_seen": False, "offloaded_layers": 0, "forbidden": []}
    pid = match.group(1)
    child_log = "\n".join(line for line in server_log.splitlines() if f"[{pid}]" in line)
    counts = [int(value) for value in re.findall(r"offloaded\s+(\d+)(?:/\d+)?\s+layers", child_log)]
    forbidden = re.findall(
        r"(?i).*(?:unsupported op|fallback to cpu|segmentation fault|fatal error|assertion .*failed|failed to allocate).*",
        child_log,
    )
    return {
        "pid": int(pid),
        "htp0_seen": "HTP0" in child_log,
        "offloaded_layers": max(counts, default=0),
        "forbidden": forbidden,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir)
    errors = []
    config = parse_ini(root / "models.ini")
    server_log = read_text(root / "server.log")
    embedding_path = "/data/local/tmp/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    reranker_path = "/data/local/tmp/models/Qwen3-Reranker-0.6B-q8_0.gguf"

    for section in ("qwen3-embedding-0.6b", "qwen3-reranker-0.6b"):
        values = config.get(section, {})
        if values.get("device") != "HTP0" or values.get("n-gpu-layers") != "99":
            errors.append(f"{section}_config_not_htp0")
        if "HTP1" in values.get("device", ""):
            errors.append(f"{section}_uses_htp1")

    embedding = json.loads(read_text(root / "embedding-response.json"))
    vector = embedding.get("data", [{}])[0].get("embedding", []) if embedding.get("data") else []
    embedding_valid = bool(vector) and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
    if not embedding_valid:
        errors.append("embedding_output_invalid")

    reranker = json.loads(read_text(root / "reranker-response.json"))
    results = reranker.get("results", [])
    reranker_valid = bool(results) and all(
        isinstance(item.get("relevance_score"), (int, float)) and math.isfinite(item["relevance_score"])
        for item in results
    )
    if not reranker_valid:
        errors.append("reranker_output_invalid")

    evidence = {
        "embedding": runtime_evidence(server_log, embedding_path),
        "reranker": runtime_evidence(server_log, reranker_path),
    }
    for name, item in evidence.items():
        if not item["htp0_seen"] or item["offloaded_layers"] <= 0:
            errors.append(f"{name}_htp0_offload_not_proven")
        if item["forbidden"]:
            errors.append(f"{name}_unsupported_or_fallback")

    shutdown = read_text(root / "shutdown.txt")
    if "graceful_shutdown_timeout=0" not in shutdown or "forced_stop_count=0" not in shutdown:
        errors.append("shutdown_not_clean")
    if read_text(root / "deployment-processes-after.txt").strip():
        errors.append("deployment_processes_remain")

    metrics = {
        "pass": not errors,
        "errors": errors,
        "embedding_dimension": len(vector),
        "embedding_output_valid": embedding_valid,
        "reranker_result_count": len(results),
        "reranker_output_valid": reranker_valid,
        "runtime_evidence": evidence,
        "config": config,
    }
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"embedding_dimension={len(vector)}")
    print(f"reranker_result_count={len(results)}")
    print("validation_errors=" + (",".join(errors) if errors else "none"))
    print(f"AUX_NPU_SMOKE_PASS={int(not errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
