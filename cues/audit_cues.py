"""Audit the cue bank: word-count parity and banned content.
Usage: ../scenarios/.venv/Scripts/python audit_cues.py"""
import json, re, statistics
cues = json.load(open("cue_bank.json", encoding="utf-8"))
BANNED = [r"\d", r"\bpercent", r"\bfirst[- ]person\b", r"\bthird[- ]person\b", r"\bI\b", r"\bwe\b",
          r"\bhe\b", r"\bshe\b", r"\bper\b", r"\baverage\b", r"\bwords? long\b", r"\bfrequen",
          r"\bcomma", r"\bsemicolon", r"\bdash", r"\bexclamation", r"['\"]\w+['\"]",
          r"\bchapter", r"\bnovel\b", r"\btitle"]
counts = {}
for a, slots in cues.items():
    assert len(slots) == 4, a
    text = " ".join(slots)
    counts[a] = len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))
    hits = [b for b in BANNED if re.search(b, text)]
    print(f"{a:4} {counts[a]:3} words  banned:{hits or 'none'}")
m = statistics.mean(counts.values())
print(f"mean {m:.1f}; range {min(counts.values())}-{max(counts.values())}; "
      f"outside +/-10%: {[a for a,c in counts.items() if abs(c-m) > 0.1*m]}")
