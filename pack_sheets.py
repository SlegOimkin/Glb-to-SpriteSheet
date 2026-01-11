import argparse
import json
from pathlib import Path
from PIL import Image


def pack_pass(output_dir: Path, pass_info: dict, frame_w: int, frame_h: int, angles: list, frames: list):
    pass_name = pass_info["name"]
    pass_dir = output_dir / pass_info["dir"]

    rows = len(angles)
    cols = len(frames)

    sheet = Image.new("RGBA", (frame_w * cols, frame_h * rows), (0, 0, 0, 0))

    total = max(1, rows * cols)
    done = 0

    for ai in range(rows):
        angle_dir = pass_dir / f"angle_{ai:02d}"
        for fi in range(cols):
            img_path = angle_dir / f"frame_{fi:04d}.png"
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))
            im = Image.open(img_path).convert("RGBA")
            sheet.paste(im, (fi * frame_w, ai * frame_h))
            done += 1
            if done % max(1, total // 50) == 0 or done == total:
                print(f"###PROGRESS {done / total:.6f}")

    out_png = output_dir / f"{pass_name}.png"
    sheet.save(out_png)

    meta = {
        "pass": pass_name,
        "sheet": str(out_png.name),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "cols": cols,
        "rows": rows,
        "angles": angles,
        "frames": frames,
    }
    out_json = output_dir / f"{pass_name}.json"
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_w, frame_h = manifest["frame_size"]
    angles = manifest["angles"]
    frames = manifest["frames"]

    passes = manifest.get("passes", [])
    for i, p in enumerate(passes):
        print(f"Packing: {p['name']} ({i+1}/{len(passes)})")
        pack_pass(output_dir, p, int(frame_w), int(frame_h), angles, frames)

    print("###PROGRESS 1.0")
    print("Done.")


if __name__ == "__main__":
    main()
