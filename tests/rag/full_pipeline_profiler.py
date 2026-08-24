#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path


def read_text(path):
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def parse_key_values(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))


def int_value(values, key, default=0):
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError):
        return default


def sum_fields(lines, prefix, names):
    result = {name: 0 for name in names}
    matched = 0
    for line in lines:
        if prefix not in line:
            continue
        matched += 1
        values = parse_key_values(line)
        for name in names:
            result[name] += int_value(values, name)
    result["records"] = matched
    return result


def parse_status(path):
    values = {}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"VmRSS", "VmHWM", "VmPeak", "VmSize"}:
            match = re.search(r"(\d+)", value)
            if match:
                values[key + "_kB"] = int(match.group(1))
    return values


def parse_faults(path):
    text = read_text(path).strip()
    if not text or ") " not in text:
        return {}
    fields = text.rsplit(") ", 1)[1].split()
    if len(fields) < 10:
        return {}
    return {"minor_faults": int(fields[7]), "major_faults": int(fields[9])}


def parse_memory(path):
    values = {}
    for line in read_text(path).splitlines():
        match = re.match(r"(MemAvailable|SwapFree):\s+(\d+)", line)
        if match:
            values[match.group(1) + "_kB"] = int(match.group(2))
    return values


def parse_timing(path):
    values = {}
    for line in read_text(path).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "http_code":
            values[key] = int(value)
        elif key in {"time_total_s"}:
            values[key] = float(value)
        elif key in {"size_download", "wall_ns_start", "wall_ns_end"}:
            values[key] = int(value)
    return values


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
            continue
        if current and "=" in line:
            key, value = line.split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def model_runtime_evidence(server_log, model_path):
    load_match = re.search(
        r"\[(\d+)\].*load_model: loading model '" + re.escape(model_path) + r"'",
        server_log,
    )
    if not load_match:
        return {"pid": None, "htp0_seen": False, "offloaded_layers": 0}
    pid = load_match.group(1)
    child_lines = [line for line in server_log.splitlines() if f"[{pid}]" in line]
    child_log = "\n".join(child_lines)
    layer_counts = [
        int(value)
        for value in re.findall(r"offloaded\s+(\d+)(?:/\d+)?\s+layers", child_log)
    ]
    return {
        "pid": int(pid),
        "htp0_seen": "HTP0" in child_log,
        "offloaded_layers": max(layer_counts, default=0),
    }


def summarize(args):
    result_dir = Path(args.result_dir)
    response = json.loads(read_text(result_dir / "response.json"))
    request = json.loads(read_text(result_dir / "request.json"))
    server_log_full = read_text(result_dir / "server.log")
    all_lines = server_log_full.splitlines()

    formal_start_path = result_dir / "server-formal-start-line.txt"
    if formal_start_path.exists():
        try:
            formal_start_line = int(read_text(formal_start_path).strip())
        except ValueError:
            formal_start_line = 0
        formal_start_line = max(0, min(formal_start_line, len(all_lines)))
        lines = all_lines[formal_start_line:]
    else:
        # Backward compatibility for old result directories.
        lines = all_lines

    server_log = "\n".join(lines)
    timing = parse_timing(result_dir / "timing.txt")

    restore_positions = [
        int(value)
        for value in re.findall(r"HANDOFF_NATIVE_RESTORE_DONE:.*?next_position=(\d+)", server_log)
    ]
    raw_candidates = response.get("generation_candidates", [])
    candidates = []
    for index, candidate in enumerate(raw_candidates):
        content = candidate.get("content", "")
        candidates.append({
            "candidate_index": candidate.get("candidate_index", index),
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "evaluated_tokens": restore_positions[index] if index < len(restore_positions) else None,
            "predicted_tokens": candidate.get("generated_tokens"),
            "success": candidate.get("success"),
            "stop_reason": candidate.get("stop_reason"),
            "prefill_success": candidate.get("prefill_success"),
            "producer_backend": candidate.get("generation_handoff_producer_backend"),
        })

    scheduler = response.get("scheduler_metrics", [])
    task_sums = {}
    for record in scheduler:
        task_type = record.get("task_type", "unknown")
        task_sums[task_type] = task_sums.get(task_type, 0) + int(record.get("execution_ms", 0))
    stage = response.get("stage_metrics", {})
    pipeline = {
        "e2e_ms": round(timing.get("time_total_s", 0.0) * 1000.0, 3),
        "total_ms": stage.get("total_ms"),
        "query_expansion_ms": task_sums.get("query_expansion", 0),
        "embedding_ms": task_sums.get("document_embedding", 0) + task_sums.get("query_embedding", 0),
        "document_embedding_ms": task_sums.get("document_embedding", 0),
        "query_embedding_ms": task_sums.get("query_embedding", 0),
        "searching_ms": task_sums.get("vector_search", 0) + task_sums.get("retrieval_merge", 0),
        "reranking_ms": task_sums.get("reranking", 0),
        "generation_ms": stage.get("generation_ms", task_sums.get("generation_prefill", 0) + task_sums.get("generation_decode", 0)),
        "npu_prefill_ms": stage.get("generation_prefill_ms", task_sums.get("generation_prefill", 0)),
        "cpu_decode_ms": stage.get("generation_decode_ms", task_sums.get("generation_decode", 0)),
    }

    config_records = []
    for line in lines:
        if "GENERATION_DECODE_CONTEXT_CONFIG:" in line:
            values = parse_key_values(line)
            config_records.append({
                "layer_chunks": int_value(values, "layer_chunks", -1),
                "cluster_attn": int_value(values, "cluster_attn", -1),
            })

    rebuild_records = []
    for line in lines:
        if "KV_HANDOFF_REBUILD_DONE:" in line:
            values = parse_key_values(line)
            rebuild_records.append({
                "ok": int_value(values, "ok"),
                "cells": int_value(values, "cells"),
                "clusters": int_value(values, "clusters"),
                "resident": int_value(values, "resident"),
            })

    layer = sum_fields(lines, "LAYER_CHUNK_SUMMARY:", [
        "setup_count", "active_setup_count", "switch_count", "prefetch_count",
        "boundaries_found", "boundaries_expected",
    ])
    madvise = sum_fields(lines, "LAYER_MADVISE_SUMMARY:", [
        "willneed_calls", "willneed_bytes", "willneed_failures",
        "pageout_calls", "pageout_bytes", "pageout_failures",
        "cold_calls", "cold_bytes", "cold_failures",
        "other_calls", "other_bytes", "other_failures",
    ])
    pager = sum_fields(lines, "KV_PAGER_SUMMARY:", [
        "read_calls", "read_bytes", "write_calls", "write_bytes", "metadata_errors",
    ])
    cluster = sum_fields(lines, "KV_CLUSTER_SUMMARY:", [
        "init_count", "eviction_count", "page_in_count", "handoff_rebuild_count",
        "handoff_rebuild_failures", "metadata_errors",
    ])
    pager_cleanup = []
    for line in lines:
        if "KV_PAGER_CLEANUP:" in line:
            values = parse_key_values(line)
            pager_cleanup.append({
                "cleanup_failures": int_value(values, "cleanup_failures"),
                "files_remaining": int_value(values, "files_remaining"),
            })

    shutdown = parse_key_values(read_text(result_dir / "shutdown.txt"))
    model_config = parse_ini(result_dir / "models.ini")
    embedding_model = "/data/local/tmp/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    reranker_model = "/data/local/tmp/models/Qwen3-Reranker-0.6B-q8_0.gguf"
    auxiliary_placement = {
        "request": {
            "document_embedding_backend": request.get("document_embedding_backend"),
            "query_embedding_backend": request.get("query_embedding_backend"),
            "reranking_backend": request.get("reranking_backend"),
            "query_expansion_backend": request.get("query_expansion_backend"),
        },
        "embedding_config": model_config.get("qwen3-embedding-0.6b", {}),
        "reranker_config": model_config.get("qwen3-reranker-0.6b", {}),
        "query_expansion_config": model_config.get("qwen3-0.6b-base", {}),
        "embedding_runtime": model_runtime_evidence(server_log_full, embedding_model),
        "reranker_runtime": model_runtime_evidence(server_log_full, reranker_model),
    }
    metadata_error_events = sum(1 for line in lines if "METADATA_ERROR:" in line)
    expected_layer = 9 if args.mode in {"layer", "full"} else 0
    expected_cluster = 1 if args.mode in {"kv", "full"} else 0
    errors = []

    if timing.get("http_code") != 200:
        errors.append("http_not_200")
    for key in {
        "document_embedding_backend", "query_embedding_backend", "reranking_backend"
    }:
        if auxiliary_placement["request"].get(key) != "npu":
            errors.append(f"{key}_request_not_npu")
    if auxiliary_placement["request"].get("query_expansion_backend") != "cpu":
        errors.append("query_expansion_backend_changed")
    for name in {"embedding_config", "reranker_config"}:
        config = auxiliary_placement[name]
        if config.get("device") != "HTP0" or config.get("n-gpu-layers") != "99":
            errors.append(f"{name}_not_htp0")
        if "HTP1" in config.get("device", ""):
            errors.append(f"{name}_uses_htp1")
    qe_config = auxiliary_placement["query_expansion_config"]
    if qe_config.get("n-gpu-layers") != "0" or "device" in qe_config:
        errors.append("query_expansion_placement_changed")
    for name in {"embedding_runtime", "reranker_runtime"}:
        evidence = auxiliary_placement[name]
        if not evidence["htp0_seen"] or evidence["offloaded_layers"] <= 0:
            errors.append(f"{name}_htp0_offload_not_proven")
    if len(candidates) != 4:
        errors.append("candidate_count_not_4")
    if len(restore_positions) != 4:
        errors.append("evaluated_token_count_missing")
    for candidate in candidates:
        idx = candidate["candidate_index"]
        if candidate["success"] is not True:
            errors.append(f"candidate_{idx}_failed")
        if candidate["prefill_success"] is not True:
            errors.append(f"candidate_{idx}_prefill_failed")
        if candidate["producer_backend"] != "npu":
            errors.append(f"candidate_{idx}_producer_not_npu")
        if candidate["predicted_tokens"] != args.max_tokens:
            errors.append(f"candidate_{idx}_predicted_mismatch")
        if not candidate["content"].strip():
            errors.append(f"candidate_{idx}_empty_content")
        if candidate["stop_reason"] != "limit":
            errors.append(f"candidate_{idx}_unexpected_stop_reason")
    if len(config_records) != 4:
        errors.append("decode_context_config_count_not_4")
    for record in config_records:
        if record["layer_chunks"] != expected_layer or record["cluster_attn"] != expected_cluster:
            errors.append("decode_context_config_mismatch")
            break
    if expected_cluster:
        if len(rebuild_records) != 4:
            errors.append("kv_rebuild_count_not_4")
        if any(record["ok"] != 1 or record["clusters"] <= 0 for record in rebuild_records):
            errors.append("kv_rebuild_invalid")
    if expected_layer:
        if layer["switch_count"] <= 0 or layer["prefetch_count"] <= 0:
            errors.append("layer_chunk_activity_missing")
        if madvise["willneed_calls"] <= 0 or (madvise["pageout_calls"] + madvise["cold_calls"]) <= 0:
            errors.append("layer_madvise_activity_missing")
    if madvise["willneed_failures"] + madvise["pageout_failures"] + madvise["cold_failures"] > 0:
        errors.append("layer_madvise_failures")
    if metadata_error_events + pager["metadata_errors"] + cluster["metadata_errors"] > 0:
        errors.append("metadata_errors")
    if any(item["cleanup_failures"] or item["files_remaining"] for item in pager_cleanup):
        errors.append("pager_cleanup_summary_failed")
    if read_text(result_dir / "pager-files-after.txt").strip():
        errors.append("pager_files_remain")
    if int_value(shutdown, "graceful_shutdown_timeout") != 0:
        errors.append("graceful_shutdown_timeout")
    if int_value(shutdown, "forced_stop_count") != 0:
        errors.append("forced_stop_used")
    if read_text(result_dir / "deployment-processes-after.txt").strip():
        errors.append("deployment_processes_remain")

    before_faults = parse_faults(result_dir / "decode-stat-before.txt")
    after_faults = parse_faults(result_dir / "decode-stat-after.txt")
    fault_delta = {}
    for key in {"minor_faults", "major_faults"}:
        if key in before_faults and key in after_faults:
            fault_delta[key] = after_faults[key] - before_faults[key]

    metrics = {
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "run_valid": not errors,
        "validation_errors": errors,
        "timing": timing,
        "pipeline": pipeline,
        "candidates": candidates,
        "decode_context_configs": config_records,
        "kv": {
            "rebuild_records": rebuild_records,
            "rebuild_count": len(rebuild_records),
            "max_clusters": max((record["clusters"] for record in rebuild_records), default=0),
            "cluster_summary": cluster,
            "pager_summary": pager,
            "pager_cleanup": pager_cleanup,
            "metadata_error_events": metadata_error_events,
        },
        "layer": {"chunk_summary": layer, "madvise_summary": madvise},
        "memory": {
            "before": parse_memory(result_dir / "memory-before.txt"),
            "after": parse_memory(result_dir / "memory-after.txt"),
            "decode_status_before": parse_status(result_dir / "decode-status-before.txt"),
            "decode_status_after": parse_status(result_dir / "decode-status-after.txt"),
            "fault_delta": fault_delta,
        },
        "shutdown": {
            "graceful_shutdown_timeout": int_value(shutdown, "graceful_shutdown_timeout"),
            "forced_stop_count": int_value(shutdown, "forced_stop_count"),
        },
        "auxiliary_placement": auxiliary_placement,
    }

    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_lines = [
        f"mode={args.mode}",
        f"run_valid={int(metrics['run_valid'])}",
        f"candidate_count={len(candidates)}",
        f"http_code={timing.get('http_code', 0)}",
        f"e2e_ms={pipeline['e2e_ms']}",
        f"kv_rebuild_count={len(rebuild_records)}",
        f"kv_max_clusters={metrics['kv']['max_clusters']}",
        f"metadata_errors={metadata_error_events + pager['metadata_errors'] + cluster['metadata_errors']}",
        f"layer_switch_count={layer['switch_count']}",
        f"layer_prefetch_count={layer['prefetch_count']}",
        f"willneed_calls={madvise['willneed_calls']}",
        f"pageout_calls={madvise['pageout_calls']}",
        f"cold_calls={madvise['cold_calls']}",
        f"graceful_shutdown_timeout={metrics['shutdown']['graceful_shutdown_timeout']}",
        f"forced_stop_count={metrics['shutdown']['forced_stop_count']}",
        f"embedding_htp0_offloaded_layers={auxiliary_placement['embedding_runtime']['offloaded_layers']}",
        f"reranker_htp0_offloaded_layers={auxiliary_placement['reranker_runtime']['offloaded_layers']}",
        "validation_errors=" + (",".join(errors) if errors else "none"),
    ]
    Path(args.text_output).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(" ".join(summary_lines[:5]))
    if errors:
        print("validation_errors=" + ",".join(errors))
        return 1
    return 0


def compare(args):
    modes = ["baseline", "kv", "layer", "full"]
    roots = {mode: Path(getattr(args, mode)) for mode in modes}
    metrics = {mode: json.loads(read_text(roots[mode] / "metrics.json")) for mode in modes}
    candidates = []
    all_identical = True
    for index in range(4):
        row = {"candidate_index": index, "modes": {}}
        values = []
        for mode in modes:
            candidate = metrics[mode]["candidates"][index]
            selected = {
                "content": candidate["content"],
                "content_sha256": candidate["content_sha256"],
                "evaluated_tokens": candidate["evaluated_tokens"],
                "predicted_tokens": candidate["predicted_tokens"],
                "success": candidate["success"],
                "stop_reason": candidate["stop_reason"],
            }
            row["modes"][mode] = selected
            values.append(selected)
        row["identical"] = all(value == values[0] for value in values[1:])
        all_identical = all_identical and row["identical"]
        candidates.append(row)

    all_runs_valid = all(metrics[mode].get("run_valid") for mode in modes)
    placement_keys = {
        "request", "embedding_config", "reranker_config", "query_expansion_config"
    }
    placement_signature = {
        mode: {
            key: metrics[mode].get("auxiliary_placement", {}).get(key)
            for key in placement_keys
        }
        for mode in modes
    }
    auxiliary_placement_identical = all(
        placement_signature[mode] == placement_signature["baseline"]
        for mode in modes[1:]
    )
    result = {
        "all_candidates_identical": all_identical,
        "all_runs_valid": all_runs_valid,
        "auxiliary_placement_identical": auxiliary_placement_identical,
        "ready": all_identical and all_runs_valid and auxiliary_placement_identical,
        "candidates": candidates,
        "result_dirs": {mode: str(roots[mode]) for mode in modes},
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = []
    for row in candidates:
        index = row["candidate_index"]
        lines.append(f"candidate_{index}:")
        for mode in modes:
            value = row["modes"][mode]
            lines.append(
                f"  {mode}_sha={value['content_sha256']} evaluated={value['evaluated_tokens']} "
                f"predicted={value['predicted_tokens']} success={int(bool(value['success']))} "
                f"stop_reason={value['stop_reason']}"
            )
        lines.append(f"  identical={int(row['identical'])}")
    lines.append(f"ALL_CANDIDATES_IDENTICAL={int(all_identical)}")
    lines.append(f"ALL_RUNS_VALID={int(all_runs_valid)}")
    lines.append(f"AUXILIARY_PLACEMENT_IDENTICAL={int(auxiliary_placement_identical)}")
    lines.append(f"PROFILER_FULL_PIPELINE_READY={int(result['ready'])}")
    Path(args.text_output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-4:]))
    return 0 if result["ready"] else 1


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--result-dir", required=True)
    summarize_parser.add_argument("--mode", choices=["baseline", "kv", "layer", "full"], required=True)
    summarize_parser.add_argument("--max-tokens", type=int, required=True)
    summarize_parser.add_argument("--output", required=True)
    summarize_parser.add_argument("--text-output", required=True)
    summarize_parser.set_defaults(func=summarize)

    compare_parser = subparsers.add_parser("compare")
    for mode in ["baseline", "kv", "layer", "full"]:
        compare_parser.add_argument(f"--{mode}", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--text-output", required=True)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
