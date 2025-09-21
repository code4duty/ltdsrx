import json
import sys
from pathlib import Path

def fix_file(input_path, output_path):
    fixed = []
    bad_lines = 0

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

        # Detect JSON array vs JSONL
        if text.startswith("["):
            examples = json.loads(text)
        else:
            examples = []
            for i, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    examples.append(ex)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping bad line {i} in {input_path}: {e}")
                    bad_lines += 1

    for ex in examples:
        for m in ex.get("mentions", []):
            if isinstance(m.get("concepts"), dict):
                new_concepts = [{"id": cid, "preferred": pref}
                                for cid, pref in m["concepts"].items()]
                m["concepts"] = new_concepts
        fixed.append(ex)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in fixed:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Fixed {input_path} → {output_path} ({len(fixed)} examples, {bad_lines} bad lines skipped)")


if __name__ == "__main__":
    data_dir = Path("data")
    for fname in ["train.jsonl", "dev.jsonl"]:
        in_path = data_dir / fname
        out_path = data_dir / fname.replace(".jsonl", "_fixed.jsonl")
        if in_path.exists():
            fix_file(in_path, out_path)
        else:
            print(f"⚠️ Skipping {in_path}, file not found")