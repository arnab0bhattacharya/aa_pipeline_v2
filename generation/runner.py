"""OpenRouter generation runner for the DSH revision (pilot + full run).

Inputs (read-only)
------------------
prompts/call_manifest.csv     2,500 calls: call_id, prompt_id, arm, model_slot, seed, prompt_sha256
prompts/imitation/*.md        prompt files; only the BODY (after the front matter) is sent
prompts/control/C###.md       blind control prompts (the runner never reads control_key.csv)
generation/models.json        frozen slot -> model + pinned provider + decoding settings

What every call sends
---------------------
    one user message = the prompt body (hash checked against the manifest before sending)
    temperature / top_p / top_k / max_tokens from models.json (slot may override max_tokens)
    seed from the manifest
    reasoning setting per slot (off where the model allows it)
    provider = {only: [pinned tag], allow_fallbacks: false, require_parameters: true}

Acceptance
----------
    First completion is accepted. ONE content retry is allowed only if the output is empty,
    looks like a refusal, or has fewer than min_words words. The retry is final either way
    (status "flagged" if it still fails). Transport failures (network, 429, 5xx) are retried
    with backoff; they produce no text and do not count as a content attempt.

Logging (append-only, outputs/<run>/calls.jsonl)
-------------------------------------------------
    "attempt" record per completion: full request (incl. prompt), raw response, served
    provider/model, usage (incl. reasoning tokens, cost), checks, output hash.
    "transport_error" record per failed HTTP try.
    "final" record per call: accepted | flagged | error. Accepted/flagged calls are skipped
    on restart (errors are retried), so the run can resume at any point.
    Accepted text is also written to outputs/<run>/texts/<call_id>.txt.

Drift guard: if more than drift_halt_fraction of finished calls were served by a provider or
model other than the pin (checked after 50 calls), the run halts.

Usage (from DSH/, with OPENROUTER_API_KEY set)
----------------------------------------------
    python generation/runner.py check              key, credit, pinned endpoints still valid
    python generation/runner.py pilot              32 pilot calls  -> outputs/pilot
    python generation/runner.py report pilot       pilot summary + seed-repeat check
    python generation/runner.py run [--limit N]    full 2,500 calls -> outputs/main
    python generation/runner.py status main        progress of a run
"""
import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
GEN = ROOT / "generation"
OUTPUTS = GEN / "outputs"
API = "https://openrouter.ai/api/v1"

CFG = json.loads((GEN / "models.json").read_text(encoding="utf-8"))
REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|i can(no|')t|i won'?t|i'?m (not able|unable)|as an ai)", re.I)
PILOT_AUTHORS = ["CD", "JC"]  # first 3 surviving scenarios of each (some ids were dropped in selection)
PILOT_SEED_REPEAT = "CD_01"
NEEDED_PARAMS = {"temperature", "top_p", "top_k", "seed", "reasoning"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set.")
    return key


def headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json",
            "X-Title": "DSH stylometry revision"}


def prompt_body(prompt_id: str, arm: str) -> str:
    path = PROMPTS / ("imitation" if arm == "imitation" else "control") / f"{prompt_id}.md"
    text = path.read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)[1]
    return body[:-1] if body.endswith("\n") else body  # writer appends one newline after the hashed body


def load_manifest() -> list[dict]:
    with open(PROMPTS / "call_manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pilot_calls(manifest: list[dict]) -> list[dict]:
    ids = {pid for a in PILOT_AUTHORS
           for pid in sorted({c["prompt_id"] for c in manifest if c["author"] == a})[:3]}
    calls = [c for c in manifest if c["prompt_id"] in ids]
    for slot in CFG["slots"]:  # first control per slot in call order
        calls.append(next(c for c in manifest if c["arm"] == "control" and c["model_slot"] == slot))
    for slot in CFG["slots"]:  # same prompt, same seed, again -> seed reproducibility
        c = next(c for c in manifest if c["prompt_id"] == PILOT_SEED_REPEAT and c["model_slot"] == slot)
        calls.append({**c, "call_id": c["call_id"] + "__rep"})
    return calls


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def build_request(call: dict, body: str) -> dict:
    slot = CFG["slots"][call["model_slot"]]
    dec = CFG["decoding"]
    return {
        "model": slot["model"],
        "messages": [{"role": "user", "content": body}],
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "top_k": dec["top_k"],
        "max_tokens": slot.get("max_tokens", dec["max_tokens"]),
        "seed": int(call["seed"]),
        "reasoning": slot["reasoning"],
        "provider": {"only": [slot["provider"]], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }


def check_text(content: str) -> tuple[str, int]:
    words = len(content.split())
    if not content.strip():
        return "empty", words
    if REFUSAL.match(content[:300]):
        return "refusal", words
    if words < CFG["acceptance"]["min_words"]:
        return "short", words
    return "ok", words


class Run:
    def __init__(self, name: str, workers: int):
        self.dir = OUTPUTS / name
        (self.dir / "texts").mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "calls.jsonl"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.workers = workers
        self.finished, self.drifted = 0, 0

    def write(self, rec: dict) -> None:
        with self.lock, open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def done_ids(self) -> set:
        if not self.log.exists():
            return set()
        with open(self.log, encoding="utf-8") as f:
            # "error" finals (gave up on transport, bad pin) are retried on the next start
            return {r["call_id"] for r in map(json.loads, f)
                    if r["type"] == "final" and r["status"] in ("accepted", "flagged")}

    def post(self, call: dict, req: dict) -> dict | None:
        """One completion; transport failures retried with backoff. None = gave up."""
        for n in range(CFG["acceptance"]["max_transport_retries"] + 1):
            if self.stop.is_set():
                return None
            t0 = time.time()
            err = None
            try:
                r = requests.post(f"{API}/chat/completions", headers=headers(), json=req, timeout=600)
                data = r.json() if r.content else {}
                if r.status_code == 200 and "error" not in data and data.get("choices"):
                    data["_latency_s"] = round(time.time() - t0, 2)
                    return data
                err = {"http": r.status_code, "body": data}
                if r.status_code in (400, 401, 402, 403, 404):  # not transient: config/credit problem
                    self.write({"type": "transport_error", "call_id": call["call_id"], "ts": now(),
                                "try": n, "error": err, "fatal": True})
                    return None
            except (requests.RequestException, ValueError) as e:
                err = {"exception": repr(e)}
            self.write({"type": "transport_error", "call_id": call["call_id"], "ts": now(), "try": n, "error": err})
            time.sleep(min(60, 2 ** (n + 1)))
        return None

    def do_call(self, call: dict) -> None:
        if self.stop.is_set():
            return
        slot = CFG["slots"][call["model_slot"]]
        body = prompt_body(call["prompt_id"], call["arm"])
        if sha256(body) != call["prompt_sha256"]:
            self.write({"type": "final", "call_id": call["call_id"], "ts": now(), "status": "error",
                        "reason": "prompt hash mismatch"})
            return
        req = build_request(call, body)
        status, attempt, content = "error", 0, ""
        for attempt in range(1, CFG["acceptance"]["max_content_retries"] + 2):
            data = self.post(call, req)
            if data is None:
                status = "error"
                break
            msg = data["choices"][0].get("message", {})
            content = msg.get("content") or ""
            check, words = check_text(content)
            usage = data.get("usage", {})
            served_provider, served_model = data.get("provider", ""), data.get("model", "")
            drift = (norm(served_provider) != norm(slot["provider"].split("/")[0])
                     or not served_model.startswith(slot["model"]))
            self.write({
                "type": "attempt", "call_id": call["call_id"], "ts": now(), "attempt": attempt,
                "prompt_id": call["prompt_id"], "arm": call["arm"], "model_slot": call["model_slot"],
                "prompt_sha256": call["prompt_sha256"], "request": req,
                "pinned_provider": slot["provider"], "served_provider": served_provider,
                "served_model": served_model, "drift": drift,
                "finish_reason": data["choices"][0].get("finish_reason"),
                "native_finish_reason": data["choices"][0].get("native_finish_reason"),
                "usage": usage,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "reasoning_text": msg.get("reasoning"),
                "content": content, "output_sha256": sha256(content), "words": words, "check": check,
                "latency_s": data.get("_latency_s"), "raw_response": data,
            })
            status = "accepted" if check == "ok" else "flagged"
            if check == "ok":
                break
        if status in ("accepted", "flagged"):
            (self.dir / "texts" / f"{call['call_id']}.txt").write_text(content, encoding="utf-8")
        self.write({"type": "final", "call_id": call["call_id"], "ts": now(), "status": status,
                    "attempts": attempt, "model_slot": call["model_slot"]})
        with self.lock:
            self.finished += 1
            self.drifted += int(status != "error" and drift)
            frac = self.drifted / self.finished
            print(f"[{self.finished}] {call['call_id']}: {status}", flush=True)
            if self.finished >= 50 and frac > CFG["drift_halt_fraction"]:
                print(f"HALT: {frac:.1%} of calls served off-pin. Re-pin before continuing.", flush=True)
                self.stop.set()

    def go(self, calls: list[dict]) -> None:
        done = self.done_ids()
        todo = [c for c in calls if c["call_id"] not in done]
        print(f"{len(calls)} calls, {len(done)} already final, {len(todo)} to do -> {self.dir}")
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self.do_call, todo))


def read_log(name: str) -> list[dict]:
    path = OUTPUTS / name / "calls.jsonl"
    if not path.exists():
        sys.exit(f"No log at {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_check(_):
    r = requests.get(f"{API}/key", headers=headers(), timeout=30)
    print("key:", r.status_code, json.dumps(r.json().get("data", r.json()), indent=1))
    ok = True
    for name, slot in CFG["slots"].items():
        eps = requests.get(f"{API}/models/{slot['model']}/endpoints", timeout=30).json()["data"]["endpoints"]
        for tag in (slot["provider"], slot["backup_provider"]):
            ep = next((e for e in eps if e["tag"] == tag), None)
            if ep is None:
                print(f"{name} {tag}: NOT LISTED")
                ok &= tag != slot["provider"]
                continue
            missing = NEEDED_PARAMS - set(ep["supported_parameters"])
            role = "pin   " if tag == slot["provider"] else "backup"
            print(f"{name} {role} {tag:18} quant={ep['quantization']:8} up1d={ep.get('uptime_last_1d')} "
                  f"in=${float(ep['pricing']['prompt'])*1e6:.3f}/M out=${float(ep['pricing']['completion'])*1e6:.3f}/M "
                  f"missing={sorted(missing) or '-'}")
            ok &= not (missing and tag == slot["provider"])
    print("ALL PINS VALID" if ok else "PIN PROBLEM: fix models.json before running")


def cmd_pilot(a):
    Run("pilot", a.workers).go(pilot_calls(load_manifest()))


def cmd_run(a):
    calls = load_manifest()
    Run("main", a.workers).go(calls[: a.limit] if a.limit else calls)


def cmd_status(a):
    recs = read_log(a.run)
    finals = [r for r in recs if r["type"] == "final"]
    by = {}
    for r in finals:
        by.setdefault(r["model_slot"], {}).setdefault(r["status"], 0)
        by[r["model_slot"]][r["status"]] += 1
    print(f"{len(finals)} final calls;", sum(r["type"] == "transport_error" for r in recs), "transport errors")
    for s in sorted(by):
        print(" ", s, by[s])


def cmd_report(a):
    recs = read_log(a.run)
    att = [r for r in recs if r["type"] == "attempt"]
    last = {}
    for r in att:
        last[r["call_id"]] = r
    print(f"{'slot':4} {'n':>3} {'words(min/med/max)':>20} {'reason_tok(med)':>15} {'cost$':>8} finish / served / flags")
    for slot in CFG["slots"]:
        rs = [r for r in last.values() if r["model_slot"] == slot]
        if not rs:
            continue
        w = sorted(r["words"] for r in rs)
        rt = sorted(r["reasoning_tokens"] or 0 for r in rs)
        cost = sum((r["usage"] or {}).get("cost", 0) or 0 for r in att if r["model_slot"] == slot)
        fins = {f: sum(r["finish_reason"] == f for r in rs) for f in {r["finish_reason"] for r in rs}}
        served = {p: sum(r["served_provider"] == p for r in rs) for p in {r["served_provider"] for r in rs}}
        heading = sum(bool(re.match(r"\s*(#|\*\*|title:)", r["content"], re.I)) for r in rs)
        checks = {c: sum(r["check"] == c for r in rs) for c in {r["check"] for r in rs}}
        print(f"{slot:4} {len(rs):>3} {w[0]:>6}/{w[len(w)//2]:>5}/{w[-1]:>5} {rt[len(rt)//2]:>15} {cost:>8.4f} "
              f"{fins} {served} checks={checks} heading_like={heading}")
    print("\nSeed repeat (same prompt + seed twice; 1.00 = identical text):")
    for cid, r in last.items():
        if cid.endswith("__rep") and cid[:-5] in last:
            ratio = difflib.SequenceMatcher(None, last[cid[:-5]]["content"], r["content"]).ratio()
            print(f"  {cid[:-5]:12} similarity={ratio:.2f} identical={r['output_sha256'] == last[cid[:-5]]['output_sha256']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(f=cmd_check)
    s = sub.add_parser("pilot"); s.add_argument("--workers", type=int, default=4); s.set_defaults(f=cmd_pilot)
    s = sub.add_parser("run"); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int); s.set_defaults(f=cmd_run)
    for n, f in (("status", cmd_status), ("report", cmd_report)):
        s = sub.add_parser(n); s.add_argument("run"); s.set_defaults(f=f)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
