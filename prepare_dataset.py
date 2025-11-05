# prepare_dataset.py
import json
from pathlib import Path
from typing import List, Dict

def merge_and_format(sources: List[Path], out_path: Path):
    out = []
    for src in sources:
        # Each source might be a jsonl of {"prompt": "...", "completion":"..."}
        with src.open("r", encoding="utf8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "prompt" in rec and "completion" in rec:
                    out.append({"prompt": rec["prompt"].strip(), "completion": rec["completion"].strip()})
    with out_path.open("w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    merge_and_format([Path(s) for s in args.sources], Path(args.out))
