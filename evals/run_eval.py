r"""Run a streaming STT evaluation on learner audio.

Examples:
  # tuned learner config vs. out-of-the-box defaults
  python -m evals.run_eval --manifest evals/data/manifest.jsonl \
      --config evals/configs/default.json --config evals/configs/learner.json

  # find the turn-detection operating point: cut-offs vs. latency
  python -m evals.run_eval --manifest evals/data/manifest.jsonl \
      --config evals/configs/learner.json --sweep practice.max_turn_silence=1600,2400,3200

  # does biasing STT toward the target phrase hide real mistakes?
  ... --sweep boost_target_keyterms=false,true
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from app.grading import grade_attempt

from .metrics import (
    bootstrap_ci,
    corpus_wer,
    percentile,
    rate,
    span_cer,
    span_hit,
    speech_end_ms,
    turn_outcome,
    wer_counts,
)
from .providers import PROVIDERS, StreamResult

PRACTICE_DEFAULTS = {"min_turn_silence": 560, "max_turn_silence": 2400}


# ------------------------------------------------------------------ inputs
def load_manifest(path: Path) -> list[dict]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if "id" not in item or "audio" not in item:
            raise ValueError(f"{path}:{line_no}: every item needs 'id' and 'audio'")
        items.append(item)
    return items


def load_audio(path: Path) -> tuple[bytes, int, np.ndarray]:
    """Any WAV/FLAC/OGG -> mono 16-bit PCM at its native rate (AssemblyAI takes 8-96 kHz)."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()
    return pcm, int(sample_rate), mono


def keyterms_for(phrase: str) -> list[str]:
    """Same rule as the web app: the phrase plus its longer words."""
    clean = "".join(ch for ch in phrase if ch not in "¿?¡!.,;:\"“”«»").strip()
    words = [w for w in clean.split() if len(w) >= 4]
    return list(dict.fromkeys([clean, *words]))[:10]


def item_settings(item: dict, cfg: dict) -> tuple[dict, dict]:
    """Connection params + UpdateConfiguration for one item, mirroring the live app."""
    params = dict(cfg.get("connect_params", {}))
    if "practice" in item.get("tags", []):
        params.update(cfg.get("practice_overrides", PRACTICE_DEFAULTS))
    update: dict = {}
    if cfg.get("language_codes"):
        update["language_codes"] = cfg["language_codes"]
    if cfg.get("use_agent_context") and item.get("agent_context"):
        update["agent_context"] = item["agent_context"]
    if cfg.get("boost_target_keyterms") and item.get("target"):
        update["keyterms_prompt"] = keyterms_for(item["target"])
    return params, update


# ------------------------------------------------------------------- run
async def run_item(item: dict, cfg: dict, base_dir: Path, api_key: str) -> dict:
    pcm, sr, mono = load_audio(base_dir / item["audio"])
    params, update = item_settings(item, cfg)
    trailing = max(int(cfg.get("trailing_silence_ms", 3000)), int(params.get("max_turn_silence", 0)) + 1500)
    provider_cls = PROVIDERS[cfg.get("provider", "assemblyai")]
    provider = provider_cls(
        api_key=api_key,
        connect_params=params,
        host=cfg.get("host", "streaming.assemblyai.com"),
        chunk_ms=int(cfg.get("chunk_ms", 50)),
        trailing_silence_ms=trailing,
        realtime_factor=float(cfg.get("realtime_factor", 1.0)),  # tests only - keep 1.0 for real evals
        url=cfg.get("url"),
    )
    result: StreamResult = await provider.transcribe(pcm, sr, update or None)

    end = float(item["speech_end_s"]) * 1000 if item.get("speech_end_s") is not None else speech_end_ms(mono, sr)
    outcome = turn_outcome([t.as_dict() for t in result.turns], end)
    hyp = result.transcript
    row: dict = {
        "id": item["id"],
        "tags": ",".join(item.get("tags", [])),
        "model": result.model,
        "error": result.error,
        "reference": item.get("reference", ""),
        "hypothesis": hyp,
        "n_turns": len(result.turns),
        "speech_end_ms": end,
        **outcome,
    }
    if item.get("reference"):
        row["wer"] = wer_counts(item["reference"], hyp)
    spans = item.get("target_spans") or []
    if spans:
        row["span_hits"] = [span_hit(s, hyp) for s in spans]
        row["span_hits_accented"] = [span_hit(s, hyp, keep_accents=True) for s in spans]
        row["span_cer"] = mean(span_cer(s, hyp) for s in spans)
    if item.get("expected_language"):
        detected = next((t.language_code for t in reversed(result.turns) if t.language_code), None)
        row["lid_expected"] = item["expected_language"]
        row["lid_detected"] = detected
    if item.get("target") and item.get("label") in ("correct", "incorrect"):
        verdict = grade_attempt(hyp, item["target"], result.words).verdict
        row["label"] = item["label"]
        row["grader_verdict"] = verdict
        row["grader_accepted"] = verdict == "correct"
    return row


async def run_config(name: str, cfg: dict, items: list[dict], base_dir: Path, api_key: str, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def guarded(item: dict) -> dict:
        nonlocal done
        async with semaphore:
            row = await run_item(item, cfg, base_dir, api_key)
        done += 1
        status = f"ERROR {row['error']}" if row.get("error") else f"\"{row['hypothesis'][:70]}\""
        print(f"  [{name}] {done}/{len(items)} {row['id']}: {status}", flush=True)
        return row

    return await asyncio.gather(*(guarded(i) for i in items))


# --------------------------------------------------------------- results
def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    out: dict = {"items": len(rows), "errors": len(rows) - len(ok)}

    with_ref = [r for r in ok if "wer" in r]
    out["wer"] = corpus_wer(r["wer"] for r in with_ref)
    out["wer_ci"] = bootstrap_ci(with_ref, lambda s: corpus_wer(r["wer"] for r in s))

    cs = [r for r in with_ref if "code_switch" in r["tags"]]
    out["wer_code_switch"] = corpus_wer(r["wer"] for r in cs) if cs else None

    span_rows = [r for r in ok if "span_hits" in r]
    span_flags = [h for r in span_rows for h in r["span_hits"]]
    out["span_accuracy"] = rate(span_flags)
    out["span_accuracy_ci"] = bootstrap_ci(span_rows, lambda s: rate([h for r in s for h in r["span_hits"]]))
    out["span_accuracy_accented"] = rate([h for r in span_rows for h in r["span_hits_accented"]])

    lat = [r["eot_latency_ms"] for r in ok if r.get("eot_latency_ms") is not None]
    out["eot_p50_ms"] = percentile(lat, 50)
    out["eot_p90_ms"] = percentile(lat, 90)
    out["eot_p50_ci"] = bootstrap_ci(lat, lambda s: percentile(s, 50))

    timed = [r for r in ok if r.get("speech_end_ms") is not None]
    out["cutoff_rate"] = rate([r["premature"] > 0 for r in timed])
    out["cutoff_rate_ci"] = bootstrap_ci(timed, lambda s: rate([r["premature"] > 0 for r in s]))
    hes = [r for r in timed if "hesitation" in r["tags"]]
    out["cutoff_rate_hesitation"] = rate([r["premature"] > 0 for r in hes]) if hes else None
    out["missed_endpoint_rate"] = rate([r["missed"] for r in timed])
    tp = sum(r["tp"] for r in timed)
    fp = sum(r["premature"] for r in timed)
    fn = sum(1 for r in timed if r["missed"])
    out["eot_accuracy"] = tp / (tp + fp + fn) if (tp + fp + fn) else None

    lid = [r for r in ok if r.get("lid_expected")]
    out["lid_accuracy"] = rate([(r.get("lid_detected") or "")[:2] == r["lid_expected"][:2] for r in lid]) if lid else None

    graded = [r for r in ok if "grader_accepted" in r]
    wrong = [r for r in graded if r["label"] == "incorrect"]
    right = [r for r in graded if r["label"] == "correct"]
    out["false_accept_rate"] = rate([r["grader_accepted"] for r in wrong]) if wrong else None
    out["false_reject_rate"] = rate([not r["grader_accepted"] for r in right]) if right else None
    out["n_labelled_wrong"], out["n_labelled_right"] = len(wrong), len(right)
    return out


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _ms(v: float | None) -> str:
    return "–" if v is None else f"{v:.0f} ms"


def _ci(ci: tuple, fmt) -> str:
    lo, hi = ci if ci else (None, None)
    return "" if lo is None else f" ({fmt(lo)}–{fmt(hi)})"


METRIC_ROWS = [
    ("Target-span accuracy (the foreign words that matter)", lambda s: _pct(s["span_accuracy"]) + _ci(s["span_accuracy_ci"], _pct)),
    ("…same, accents must match too", lambda s: _pct(s["span_accuracy_accented"])),
    ("WER, all utterances", lambda s: _pct(s["wer"]) + _ci(s["wer_ci"], _pct)),
    ("WER, code-switched utterances", lambda s: _pct(s["wer_code_switch"])),
    ("Premature cut-off rate", lambda s: _pct(s["cutoff_rate"]) + _ci(s["cutoff_rate_ci"], _pct)),
    ("…on utterances with hesitation pauses", lambda s: _pct(s["cutoff_rate_hesitation"])),
    ("Missed endpoint rate", lambda s: _pct(s["missed_endpoint_rate"])),
    ("EOT accuracy TP/(TP+FP+FN)", lambda s: _pct(s["eot_accuracy"])),
    ("End of speech → final transcript, p50", lambda s: _ms(s["eot_p50_ms"]) + _ci(s["eot_p50_ci"], _ms)),
    ("End of speech → final transcript, p90", lambda s: _ms(s["eot_p90_ms"])),
    ("Language ID accuracy", lambda s: _pct(s["lid_accuracy"])),
    ("Grader false-accept rate (wrong attempt praised)", lambda s: _pct(s["false_accept_rate"]) + f" of {s['n_labelled_wrong']}"),
    ("Grader false-reject rate (right attempt rejected)", lambda s: _pct(s["false_reject_rate"]) + f" of {s['n_labelled_right']}"),
    ("Items / errors", lambda s: f"{s['items']} / {s['errors']}"),
]


def write_report(out_dir: Path, results: dict[str, tuple[dict, list[dict], dict]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results)
    lines = [
        "# Streaming STT evaluation — language tutor",
        "",
        f"_Run {datetime.now():%Y-%m-%d %H:%M}. 95% bootstrap confidence intervals over utterances in brackets._",
        "",
    ]
    n_items = max(s["items"] for s, _, _ in results.values())
    if n_items < 100:
        lines += [
            f"> **Smoke test only:** {n_items} utterances. Intervals are wide; a decision-grade eval needs "
            "a few hundred utterances per stratum from real users.",
            "",
        ]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for label, fmt in METRIC_ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(results[n][0]) for n in names) + " |")
    lines += ["", "## Configurations", ""]
    for n in names:
        lines += [f"**{n}**", "", "```json", json.dumps(results[n][2], indent=2, ensure_ascii=False), "```", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    with open(out_dir / "items.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["config", "id", "tags", "model", "reference", "hypothesis", "n_turns", "premature", "missed",
                  "eot_latency_ms", "span_cer", "lid_expected", "lid_detected", "label", "grader_verdict", "error"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for n in names:
            for row in results[n][1]:
                writer.writerow({"config": n, **row})
    return report


# -------------------------------------------------------------------- CLI
def parse_value(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def expand_sweep(name: str, cfg: dict, sweep: str | None) -> list[tuple[str, dict]]:
    if not sweep:
        return [(name, cfg)]
    key, _, values = sweep.partition("=")
    variants = []
    for raw in values.split(","):
        value = parse_value(raw.strip())
        variant = copy.deepcopy(cfg)
        if key in ("use_agent_context", "boost_target_keyterms"):
            variant[key] = value
        elif key.startswith("practice."):
            variant.setdefault("practice_overrides", dict(PRACTICE_DEFAULTS))[key.split(".", 1)[1]] = value
        else:
            variant.setdefault("connect_params", {})[key] = value
        variants.append((f"{name} · {key}={raw.strip()}", variant))
    return variants


async def evaluate(
    manifest: Path, configs: list[dict], sweep: str | None, api_key: str, concurrency: int = 4, limit: int | None = None
) -> dict[str, tuple[dict, list[dict], dict]]:
    items = load_manifest(manifest)[:limit]
    variants: list[tuple[str, dict]] = []
    for i, cfg in enumerate(configs):
        variants += expand_sweep(cfg.get("name", f"config {i + 1}"), cfg, sweep)
    results = {}
    for name, cfg in variants:
        print(f"▶ {name}: {len(items)} utterances, streamed in real time", flush=True)
        rows = await run_config(name, cfg, items, manifest.parent, api_key, concurrency)
        results[name] = (summarise(rows), rows, cfg)
    return results


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True, help="repeat to compare configs")
    parser.add_argument("--sweep", help="key=v1,v2 (connect param, practice.<param>, use_agent_context, boost_target_keyterms)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path("evals/results"))
    args = parser.parse_args(argv)

    api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    if not api_key:
        print("Set ASSEMBLYAI_API_KEY (e.g. in .env)", file=sys.stderr)
        return 2
    configs = []
    for path in args.config:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg.setdefault("name", path.stem)
        configs.append(cfg)

    results = asyncio.run(evaluate(args.manifest, configs, args.sweep, api_key, args.concurrency, args.limit))

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = write_report(out_dir, results)
    print("\n" + report.read_text(encoding="utf-8").split("## Configurations")[0])
    print(f"Full report: {report}\nPer-utterance results: {out_dir / 'items.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
