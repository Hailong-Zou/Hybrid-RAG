#!/usr/bin/env python3
# Convert a downloaded MCQ benchmark into the JSONL format the llama-conflayer
# --eval mode reads: one {"question":..., "choices":[4], "answer":<0..3>} per line.
#
# Auto-detects the common single-letter-answer MCQ shapes:
#   * OpenBookQA / ARC   (allenai/*):  {"question":{"stem":..., "choices":[
#                                       {"text":...,"label":"A".."D"}]},
#                                       "answerKey":"B"}
#   * MMLU               (hendrycks):  CSV columns `question,A,B,C,D,answer`
#   * generic JSONL:      {"question":..., "choices":[...], "answer":<idx or letter>}
#
# Usage:
#   python3 make_eval_jsonl.py INPUT [OUTPUT] [--limit N]
# INPUT is a .jsonl/.json/.csv/.tsv file; OUTPUT defaults to eval.jsonl next to it.
# answer may be a 0..3 index, a "A".."D" letter, or a choice string that exactly
# matches one of the choices.
import argparse
import csv
import json
import os
import sys


def letter_to_idx(letter):
    letter = str(letter).strip().upper()
    if letter in "ABCD":
        return ord(letter) - ord("A")
    return None


def normalize_answer(ans, choices):
    if isinstance(ans, int):
        return ans if 0 <= ans < len(choices) else None
    if isinstance(ans, float) and ans == int(ans):
        return int(ans) if 0 <= int(ans) < len(choices) else None
    s = str(ans).strip()
    idx = letter_to_idx(s)
    if idx is not None:
        return idx
    for i, c in enumerate(choices):
        if s == str(c).strip():
            return i
    return None


def detect_record(obj):
    """Return (question, choices, answer) for a raw benchmark record, or None."""
    # nested question.stem + choices[{text,label}] + answerKey  (OpenBookQA / ARC)
    if isinstance(obj.get("question"), dict):
        q = obj["question"]
        stem = q.get("stem")
        cq = q.get("choices")
        if stem and isinstance(cq, list) and cq:
            choices = [str(c["text"]) for c in cq]
            # keep the labels' original order so the answer index stays aligned
            labels = [str(c.get("label", "")).strip() for c in cq]
            answer = obj.get("answerKey")
            if answer is not None:
                idx = letter_to_idx(answer)
                if idx is not None:
                    # label may be ordered differently than A/B/C/D; remap
                    if len(labels) == 4 and all(l in "ABCD" for l in labels):
                        pos = labels.index(str(answer).strip().upper()) \
                            if str(answer).strip().upper() in labels else None
                        if pos is not None:
                            return stem, choices, pos
            idx = normalize_answer(answer, choices)
            if idx is not None:
                return stem, choices, idx
    # flat {question, choices, answer}
    if "question" in obj and "choices" in obj:
        choices = [str(c) for c in obj["choices"]]
        idx = normalize_answer(obj.get("answer", obj.get("answerKey")), choices)
        if idx is not None:
            return str(obj["question"]), choices, idx
    return None


def convert(path, out_path, limit):
    in_lines, ext = [], os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            rows = list(csv.DictReader(f, delimiter=delim))
            # MMLU: question,A,B,C,D,answer(letter)
            if "A" in rows[0] and "B" in rows[0] and "question" in rows[0]:
                for r in rows:
                    choices = [r.get("A", ""), r.get("B", ""), r.get("C", ""),
                               r.get("D", "")]
                    idx = letter_to_idx(r.get("answer", ""))
                    if idx is not None:
                        in_lines.append({"question": r["question"], "choices": choices,
                                         "answer": idx})
            else:
                sys.exit("csv: expected MMLU columns question,A,B,C,D,answer")
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0] != "{":  # skip dataset JSON prefixes
                    continue
                obj = json.loads(line)
                rec = detect_record(obj)
                if rec is not None:
                    in_lines.append({"question": rec[0], "choices": rec[1],
                                     "answer": rec[2]})

    if not in_lines:
        sys.exit("no recognizable MCQ records in %s" % path)
    if limit and len(in_lines) > limit:
        in_lines = in_lines[:limit]

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in in_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %d questions to %s" % (len(in_lines), out_path))
    print("  sample:", json.dumps(in_lines[0], ensure_ascii=False)[:160], "...")
    print("next: ./llama-conflayer-example -m model.gguf --eval %s --eval-limit 200 -t 8"
          % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = args.output or os.path.splitext(args.input)[0] + "_eval.jsonl"
    convert(args.input, out, args.limit)


if __name__ == "__main__":
    main()
