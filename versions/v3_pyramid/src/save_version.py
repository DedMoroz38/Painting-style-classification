"""Save a snapshot of the current code, trained models and results to versions/<name>/.

    python src/save_version.py v3_pyramid "500 words + 2x2 spatial pyramid"

Copies src/ and the traditional-pipeline outputs (features, vocabularies, models, results;
not outputs/deep). On macOS the copies are APFS clones (cp -c), which use no extra disk
space until a file changes. Also writes versions/<name>/NOTE.txt with the test scores.
"""
import subprocess
import sys
from datetime import datetime

from config import OUTPUT_DIR, PROJECT_DIR

VERSIONS_DIR = PROJECT_DIR / "versions"
OUTPUT_ITEMS = ["features.npz", "features_13.npz", "bovw_vocabulary_5.npy", "bovw_vocabulary_13.npy",
                "models", "models_13", "results", "results_13"]


def copy(src, dst):
    """Recursive copy; APFS clone where possible, plain copy otherwise."""
    if subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True).returncode != 0:
        subprocess.run(["cp", "-R", str(src), str(dst)], check=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    name, description = sys.argv[1], " ".join(sys.argv[2:])
    out = VERSIONS_DIR / name
    if out.exists():
        sys.exit(f"{out} already exists - pick a new name (versions are never overwritten)")
    (out / "outputs").mkdir(parents=True)

    copy(PROJECT_DIR / "src", out / "src")
    subprocess.run(["rm", "-rf", str(out / "src" / "__pycache__")])
    for item in OUTPUT_ITEMS:
        if (OUTPUT_DIR / item).exists():
            copy(OUTPUT_DIR / item, out / "outputs" / item)

    # NOTE.txt: description + the headline test score of every saved result
    lines = [f"{name}  ({datetime.now():%Y-%m-%d %H:%M})", description, ""]
    for report in sorted(out.glob("outputs/results*/*/report.txt")):
        score = next(l for l in report.read_text().splitlines() if l.startswith("TEST macro-F1"))
        lines.append(f"{report.parent.parent.name}/{report.parent.name}: {score}")
    (out / "NOTE.txt").write_text("\n".join(lines) + "\n")
    print((out / "NOTE.txt").read_text())


if __name__ == "__main__":
    main()
