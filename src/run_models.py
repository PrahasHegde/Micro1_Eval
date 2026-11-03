"""Sends every task prompt to every enabled model and stores the raw outputs.

    python src/run_models.py                        # all enabled models, both conditions, 3 samples
    python src/run_models.py --models claude-opus-5-5 --tasks H1,T2 --samples 1
    python src/run_models.py --dry-run              # print the plan, call nothing

One JSON file per (model, condition, task, sample) is written to
results/raw/<model>/<condition>/<task>__s<k>.json. Existing successful files
are skipped, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "models.json"
TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"
RAW_DIR = ROOT / "results" / "raw"
SYSTEM_PROMPT = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").strip()
CONDITIONS = ("closed_book", "open_book")


def call_anthropic(cfg: dict, prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ[cfg.get("api_key_env", "ANTHROPIC_API_KEY")], max_retries=6)
    params = dict(cfg.get("params", {}))
    max_tokens = params.pop("max_tokens", 32000)
    # Streaming avoids HTTP timeouts on long thinking turns.
    with client.messages.stream(model=cfg["model"], max_tokens=max_tokens, system=SYSTEM_PROMPT,
                                messages=[{"role": "user", "content": prompt}], **params) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    return dict(text=text, response_model=msg.model, stop_reason=msg.stop_reason,
                usage=dict(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens))


def call_openai_compat(cfg: dict, prompt: str) -> dict:
    import openai

    client = openai.OpenAI(api_key=os.environ[cfg["api_key_env"]], base_url=cfg.get("base_url"), max_retries=6,
                           timeout=900)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        **cfg.get("params", {}))
    choice = resp.choices[0]
    usage = resp.usage
    return dict(text=choice.message.content or "", response_model=resp.model, stop_reason=choice.finish_reason,
                usage=dict(input_tokens=getattr(usage, "prompt_tokens", None),
                           output_tokens=getattr(usage, "completion_tokens", None)))


def call_ollama(cfg: dict, prompt: str) -> dict:
    """Local open-weight models through Ollama's native API, which (unlike its
    OpenAI-compatible endpoint) accepts num_ctx, seed and the think switch."""
    body = dict(model=cfg["model"], stream=False,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                **cfg.get("params", {}))
    base = cfg.get("base_url") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    req = urllib.request.Request(f"{base.rstrip('/')}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.loads(r.read())
    msg = resp.get("message", {})
    return dict(text=msg.get("content", ""), thinking=msg.get("thinking"), response_model=resp.get("model"),
                stop_reason=resp.get("done_reason"),
                usage=dict(input_tokens=resp.get("prompt_eval_count"), output_tokens=resp.get("eval_count")))


PROVIDERS = {"anthropic": call_anthropic, "openai_compat": call_openai_compat, "ollama": call_ollama}


def out_path(label: str, condition: str, task_id: str, sample: int) -> Path:
    return RAW_DIR / label / condition / f"{task_id}__s{sample}.json"


def run_one(cfg: dict, task: dict, condition: str, sample: int) -> tuple[Path, str | None]:
    prompt = task["prompts"][condition]
    rec = dict(model_label=cfg["label"], provider=cfg["provider"], requested_model=cfg["model"],
               params=cfg.get("params", {}), condition=condition, task_id=task["id"], sample=sample,
               prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
               timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    t0 = time.time()
    try:
        rec.update(PROVIDERS[cfg["provider"]](cfg, prompt))
        rec["error"] = None
    except Exception as e:  # recorded and retried on the next invocation
        rec.update(text="", error=f"{type(e).__name__}: {e}")
    rec["latency_s"] = round(time.time() - t0, 2)
    path = out_path(cfg["label"], condition, task["id"], sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, rec["error"]


def already_done(path: Path) -> bool:
    if not path.exists():
        return False
    rec = json.loads(path.read_text(encoding="utf-8"))
    return not rec.get("error") and bool(rec.get("text"))


def write_manifest(models: list[dict], conditions: list[str], samples: int) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    tasks_sha = hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest()
    manifest_path = ROOT / "results" / "run_manifest.json"
    runs = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    runs.append(dict(started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit=commit,
                     tasks_sha256=tasks_sha, python=sys.version.split()[0], conditions=conditions,
                     samples=samples, models=[{k: v for k, v in m.items() if k != "enabled"} for m in models]))
    manifest_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated labels (default: all enabled in config/models.json)")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["models"]
    if args.models:
        wanted = args.models.split(",")
        models = [m for m in config if m["label"] in wanted]
        missing = set(wanted) - {m["label"] for m in models}
        if missing:
            sys.exit(f"unknown model labels: {sorted(missing)}")
    else:
        models = [m for m in config if m.get("enabled", True)]
    with TASKS_PATH.open(encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in args.tasks.split(",")]
    conditions = args.conditions.split(",")

    jobs = [(m, t, c, s) for m in models for c in conditions for t in tasks for s in range(1, args.samples + 1)]
    todo = [j for j in jobs if not already_done(out_path(j[0]["label"], j[2], j[1]["id"], j[3]))]
    print(f"{len(models)} models x {len(conditions)} conditions x {len(tasks)} tasks x {args.samples} samples "
          f"= {len(jobs)} calls ({len(todo)} still to do)")
    if args.dry_run or not todo:
        return
    for m in models:
        if m["provider"] == "ollama":
            continue
        key_env = m.get("api_key_env", "ANTHROPIC_API_KEY")
        if not os.environ.get(key_env):
            sys.exit(f"{m['label']}: environment variable {key_env} is not set (see .env.example)")
    write_manifest(models, conditions, args.samples)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *j): j for j in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            path, err = fut.result()
            failures += bool(err)
            status = f"ERROR {err[:120]}" if err else "ok"
            print(f"[{i}/{len(todo)}] {path.relative_to(RAW_DIR).as_posix()} {status}", flush=True)
    print(f"done; {failures} failed calls (re-run the same command to retry them)")


if __name__ == "__main__":
    main()
