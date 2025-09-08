import json
import sys
from pathlib import Path
def fix_file(input_path, output_path):
    fixed = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:   # skip empty lines
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping bad line {i} in {input_path}: {e}")
                continue
            for m in ex.get("mentions", []):
                if isinstance(m.get("concepts"), dict):
                    new_concepts = []
                    for cid, pref in m["concepts"].items():
                        new_concepts.append({"id": cid, "preferred": pref})
                    m["concepts"] = new_concepts
            fixed.append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in fixed:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Fixed {input_path} → {output_path} ({len(fixed)} examples)")
if __name__ == "__main__":
    data_dir = Path("data")
    for fname in ["train.jsonl", "dev.jsonl"]:
        in_path = data_dir / fname
        out_path = data_dir / fname.replace(".jsonl", "_fixed.jsonl")
        if in_path.exists():
            fix_file(in_path, out_path)
        else:
            print(f"⚠️ Skipping {in_path}, file not found")