"""Assemble the 1,000 unique generation prompts and the 2,500-call manifest.

Pairing logic
-------------
Imitation (500 unique prompts, one per scenario):
    author line  <- author of the scenario (name + life dates)
    cue          <- cues/cue_bank.json[author]  (same cue for all 50 of that author's scenarios)
    anchor       <- scenarios_final_paired.csv: anchor_id -> anchors/anchor_manifest.csv -> chunk text
                    (each of the author's 25 anchors serves exactly 2 scenarios)
    scenario     <- premise + setting of that scenario
    Every imitation prompt is sent to all 4 models -> 2,000 calls.

Control (500 unique prompts, one per scenario):
    same scenario premise + setting, with NO author, cue or anchor.
    Each control prompt is sent to ONE model. Per author, two slots get 13 controls
    and two get 12; the "13" pairs (THIRTEEN_PAIRS) are chosen so each slot gets 13
    from exactly five authors -> exactly 125 controls per slot. Within an author,
    slots are matched to scenarios by a seeded shuffle.

Models are recorded as slots M1-M4 until the four OpenRouter models are fixed.

Outputs (this folder)
---------------------
imitation/{scenario_id}.md      one file per imitation prompt
control/C###.md                 one file per control prompt; neutral id, metadata holds NO
                                author, scenario or mode (blinding). The link back lives
                                only in control_key.csv, read by the analysis, not the runner.
    Each file = YAML front matter (metadata) + '---' + the prompt body.
    ONLY THE BODY (everything after the second '---' line) is sent to a model;
    prompt_sha256 is the hash of that body.
control_key.csv                 500 rows: control id -> scenario_id, author, mode
prompt_manifest.csv             1,000 rows (control rows carry no author/scenario/mode): prompt_id, arm, author, scenario, anchor, hash, word counts
call_manifest.csv               2,500 rows: call order (seeded shuffle), model slot, per-call seed
build_info.json                 input hashes and settings, for reproducibility

Usage (from DSH/): scenarios/.venv/Scripts/python prompts/build_prompts.py
"""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "prompts"
SCENARIOS = ROOT / "scenarios" / "scenarios_final_paired.csv"
ANCHOR_MANIFEST = ROOT / "anchors" / "anchor_manifest.csv"
ANCHOR_DIR = ROOT / "anchors"
CUES = ROOT / "cues" / "cue_bank.json"

SEED = 42
MODEL_SLOTS = ["M1", "M2", "M3", "M4"]
# slot pairs receiving 13 controls, by author index (sorted author codes); each slot appears 5 times
THIRTEEN_PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2), (0, 1), (2, 3), (0, 2), (1, 3)]

AUTHORS = {
    "ACD": ("Arthur Conan Doyle", "1859–1930"),
    "CB": ("Charlotte Brontë", "1816–1855"),
    "CD": ("Charles Dickens", "1812–1870"),
    "GE": ("George Eliot", "1819–1880"),
    "HGW": ("H. G. Wells", "1866–1946"),
    "JA": ("Jane Austen", "1775–1817"),
    "JC": ("Joseph Conrad", "1857–1924"),
    "MS": ("Mary Shelley", "1797–1851"),
    "OW": ("Oscar Wilde", "1854–1900"),
    "VW": ("Virginia Woolf", "1882–1941"),
}

REQUIREMENTS = """[Requirements]
- About 1,200 words of continuous narrative prose, in paragraphs.
- Invent all names of characters and places.
- No title, headings, notes or commentary before or after the story."""

IMITATION_TEMPLATE = """Write an original piece of prose fiction in the style of {name} ({dates}).

[Notes on the author's manner (for orientation, not a checklist)]
{cue}

[Style sample]
The following passage is by {name}. It is provided only to show the author's style. Do not continue it, quote it, paraphrase it, or reuse any of its characters, names, places or events.
<<<
{anchor}
>>>

[Scenario]
{premise}
Setting: {setting}

""" + REQUIREMENTS

CONTROL_TEMPLATE = """Write an original piece of prose fiction.

[Scenario]
{premise}
Setting: {setting}

""" + REQUIREMENTS


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> int:
    return len(text.split())


def write_prompt(path: Path, meta: dict, body: str) -> None:
    front = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())
    path.write_text(f"---\n{front}\n---\n{body}\n", encoding="utf-8")


def write_body(cid: str) -> str:
    text = (OUT / "control" / f"{cid}.md").read_text(encoding="utf-8")
    return text.split("\n---\n", 1)[1]


def main():
    cues = json.loads(CUES.read_text(encoding="utf-8"))
    anchors = {r["anchor_id"]: r for r in csv.DictReader(open(ANCHOR_MANIFEST, encoding="utf-8"))}
    scenarios = list(csv.DictReader(open(SCENARIOS, encoding="utf-8")))
    assert len(scenarios) == 500, len(scenarios)

    (OUT / "imitation").mkdir(parents=True, exist_ok=True)
    (OUT / "control").mkdir(parents=True, exist_ok=True)

    prompt_rows, control_key = [], []
    authors = sorted(AUTHORS)
    slot_rng = random.Random(SEED)
    for a_idx, author in enumerate(authors):
        name, dates = AUTHORS[author]
        cue = " ".join(cues[author])
        rows = sorted((s for s in scenarios if s["author"] == author), key=lambda s: s["scenario_id"])
        assert len(rows) == 50, (author, len(rows))
        slots = [m for k, m in enumerate(MODEL_SLOTS) for _ in range(13 if k in THIRTEEN_PAIRS[a_idx] else 12)]
        slot_rng.shuffle(slots)
        for s_idx, s in enumerate(rows):
            sid = s["scenario_id"]
            anc = anchors[s["anchor_id"]]
            assert anc["author"] == author, (sid, anc["anchor_id"])
            anchor_text = (ANCHOR_DIR / author / f"{anc['chunk_id']}.txt").read_text(encoding="utf-8").strip()

            body = IMITATION_TEMPLATE.format(name=name, dates=dates, cue=cue, anchor=anchor_text,
                                             premise=s["premise"], setting=s["setting"])
            h = sha256(body)
            meta = {"prompt_id": sid, "arm": "imitation", "author": author, "scenario_id": sid,
                    "mode": s["mode"], "anchor_id": anc["anchor_id"], "anchor_chunk_id": anc["chunk_id"],
                    "anchor_work": anc["work"], "anchor_form": anc["form"], "models": MODEL_SLOTS,
                    "prompt_sha256": h}
            write_prompt(OUT / "imitation" / f"{sid}.md", meta, body)
            prompt_rows.append({**{k: meta[k] for k in ("prompt_id", "arm", "author", "scenario_id", "mode",
                                                        "anchor_id", "anchor_chunk_id", "anchor_work", "anchor_form")},
                                "model_slots": "M1;M2;M3;M4", "prompt_words": words(body),
                                "anchor_words": words(anchor_text), "prompt_sha256": h})

            slot = slots[s_idx]
            cbody = CONTROL_TEMPLATE.format(premise=s["premise"], setting=s["setting"])
            ch = sha256(cbody)
            control_key.append({"scenario_id": sid, "author": author, "mode": s["mode"],
                                "model_slot": slot, "prompt_sha256": ch, "cbody": cbody})

    # neutral control ids, assigned in a seeded random order so the id carries no author information
    random.Random(SEED + 1).shuffle(control_key)
    for n, k in enumerate(control_key, 1):
        cid = f"C{n:03d}"
        k["prompt_id"] = cid
        cmeta = {"prompt_id": cid, "arm": "control", "models": [k["model_slot"]],
                 "prompt_sha256": k["prompt_sha256"]}
        write_prompt(OUT / "control" / f"{cid}.md", cmeta, k.pop("cbody"))
        prompt_rows.append({"prompt_id": cid, "arm": "control", "author": "", "scenario_id": "",
                            "mode": "", "anchor_id": "", "anchor_chunk_id": "", "anchor_work": "",
                            "anchor_form": "", "model_slots": k["model_slot"],
                            "prompt_words": words(write_body(cid)), "anchor_words": 0,
                            "prompt_sha256": k["prompt_sha256"]})
    with open(OUT / "control_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "scenario_id", "author", "mode", "model_slot", "prompt_sha256"])
        w.writeheader()
        w.writerows(sorted(control_key, key=lambda k: k["prompt_id"]))

    with open(OUT / "prompt_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prompt_rows[0]))
        w.writeheader()
        w.writerows(prompt_rows)

    calls = []
    for p in prompt_rows:
        for slot in p["model_slots"].split(";"):
            calls.append({"prompt_id": p["prompt_id"], "arm": p["arm"], "author": p["author"],
                          "scenario_id": p["scenario_id"], "model_slot": slot,
                          "prompt_sha256": p["prompt_sha256"]})
    rng = random.Random(SEED)
    rng.shuffle(calls)
    for i, c in enumerate(calls, 1):
        c["call_order"] = i
        c["call_id"] = f"{c['prompt_id']}__{c['model_slot']}"
        c["seed"] = int(sha256(f"{c['call_id']}|{SEED}")[:8], 16)  # deterministic per call
    fields = ["call_order", "call_id", "prompt_id", "arm", "author", "scenario_id", "model_slot",
              "seed", "prompt_sha256"]
    with open(OUT / "call_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: c[k] for k in fields} for c in calls)

    info = {"seed": SEED, "model_slots": MODEL_SLOTS,
            "inputs_sha256": {"scenarios_final_paired.csv": file_sha256(SCENARIOS),
                              "anchor_manifest.csv": file_sha256(ANCHOR_MANIFEST),
                              "cue_bank.json": file_sha256(CUES)},
            "imitation_template": IMITATION_TEMPLATE, "control_template": CONTROL_TEMPLATE,
            "n_prompts": len(prompt_rows), "n_calls": len(calls)}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(prompt_rows)} prompts, {len(calls)} calls")


if __name__ == "__main__":
    main()
