"""
Merge post-dedup structures.json from each conformer's full-pipeline run
(test_mini_csp_full_pipeline_timing.py output) into a single consolidated
JSON file, tagging each structure with its source conformer.

Run with:
    python merge_pipeline_structures.py
"""
import os, json, glob

work_root = "/home/xchen/Test/Genarris/test/lbwl-1/csp_16conf_grid"
out_path = os.path.join(work_root, "merged_structures.json")

conf_dirs = sorted(
    glob.glob(os.path.join(work_root, "conf_*")),
    key=lambda p: int(os.path.basename(p).split("_")[1]),
)

merged = {}
per_conf_counts = {}
collisions = 0

for conf_dir in conf_dirs:
    conf_id = os.path.basename(conf_dir)
    struct_path = os.path.join(conf_dir, "structures", "dedup", "structures.json")
    if not os.path.exists(struct_path):
        print(f"  [SKIP] {conf_id}: no dedup structures.json found")
        continue

    with open(struct_path) as f:
        structs = json.load(f)

    per_conf_counts[conf_id] = len(structs)
    for key, entry in structs.items():
        merged_key = f"{conf_id}__{key}"
        if merged_key in merged:
            collisions += 1
        entry.setdefault("info", {})
        entry["info"]["conformer_id"] = conf_id
        entry["info"]["conformer_mol_path"] = os.path.join(conf_dir, "mol.xyz")
        merged[merged_key] = entry

with open(out_path, "w") as f:
    json.dump(merged, f)

print("Per-conformer structure counts (post-dedup):")
for conf_id, n in per_conf_counts.items():
    print(f"  {conf_id:<10} {n}")
print(f"\nTotal structures merged: {len(merged)}  (key collisions: {collisions})")
print(f"Written to: {out_path}")
