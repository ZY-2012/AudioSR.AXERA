import sys
from pathlib import Path

THRESHOLD = 0.98
FINAL_NAMES = {
    "vae_encoder": {"mean", "logvar"},
    "vae_decoder": {"mel"},
    "vocoder": {"wav"},
    "unet": {"v"},
}


def parse_table(path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if "│" not in line:
            continue
        parts = [p.strip() for p in line.strip().strip("│").split("│")]
        if len(parts) < 8:
            continue
        tensor = parts[2]
        try:
            cos = float(parts[6])
            mse = float(parts[7])
        except ValueError:
            continue
        rows.append((tensor, cos, mse, parts[1]))
    return rows


def main():
    name = sys.argv[1]
    base = Path(__file__).resolve().parents[1]
    path = base / "build" / "compile" / f"{name}_work" / "quant" / "debug" / "precision_analysis_table.txt"
    print(f"=== precision gate: {name} (threshold {THRESHOLD:.2f}) ===")
    if not path.exists():
        print(f"FAIL: precision table not found: {path}")
        return 1
    rows = parse_table(path)
    if not rows:
        print("FAIL: no parsable precision rows")
        return 1
    expected = None
    for key, names in FINAL_NAMES.items():
        if name.startswith(key):
            expected = names
            break
    selected = [r for r in rows if r[0] in expected] if expected else rows[-1:]
    if not selected:
        print(f"WARN: no final output names {sorted(expected)}, using last row")
        selected = rows[-1:]
    failed = False
    for tensor, cos, mse, typ in selected:
        ok = cos >= THRESHOLD
        failed = failed or not ok
        print(f"{tensor:10s} cos={cos:.6f} mse={mse:.6g} type={typ} {'PASS' if ok else 'FAIL'}")
    if failed:
        print("GATE FAILED: stop quantization; change config/model before next compile.")
        return 2
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
