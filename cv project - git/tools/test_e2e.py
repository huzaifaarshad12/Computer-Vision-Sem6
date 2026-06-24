"""
End-to-end test harness — exercises the running server exactly like the React UI.
It replicates the frontend's verdict logic so each row shows what a USER would see.

Start the server (python server.py) in another terminal, then:  python tools/test_e2e.py
"""
import io
import os
import sys

import requests
from PIL import Image

BASE = "http://localhost:8000"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# key the UI sends -> key the server returns
RETURN_KEY = {"classify": "classify", "detect": "detect",
              "segment": "segment", "explain": "gradcam"}

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def ui_verdict(r):
    """Replicate the React isFire computation."""
    fire = (
        (r.get("classify") and r["classify"]["label"] == "fire_smoke") or
        (r.get("gradcam") and r["gradcam"]["label"] == "fire_smoke") or
        (r.get("detect") and r["detect"]["count"] > 0) or
        (r.get("segment") and r["segment"]["count"] > 0)
    )
    return "FIRE/SMOKE DETECTED" if fire else "ALL CLEAR"


def analyze(img_path, modules):
    with open(img_path, "rb") as f:
        files = {"file": ("img.jpg", f, "image/jpeg")}
        data = {"modules": ",".join(modules)}
        return requests.post(f"{BASE}/api/analyze", files=files, data=data, timeout=120)


def find_normal():
    d = os.path.join(HERE, "dataset/classification/test/normal")
    if os.path.isdir(d):
        return os.path.join(d, sorted(os.listdir(d))[0])
    return None


def main():
    print("\n=== 1. Connectivity ===")
    s = requests.get(f"{BASE}/api/status", timeout=10).json()
    check("/api/status returns 3 model flags", set(s) == {"classifier", "detector", "segmenter"}, str(s))
    check("all 3 models loaded", all(s.values()), str(s))
    page = requests.get(f"{BASE}/", timeout=10)
    check("/ serves React page", page.status_code == 200 and 'id="root"' in page.text)

    fire = os.path.join(HERE, "Fire-Segmentation-Dataset-main/images/1000.jpg")
    normal = find_normal()

    # (image label, path, expected verdict)
    images = [("FIRE", fire, "FIRE/SMOKE DETECTED")]
    if normal:
        images.append(("NORMAL", normal, "ALL CLEAR"))

    module_sets = [
        ["classify", "detect", "segment", "explain"],
        ["classify"], ["detect"], ["segment"], ["explain"],
        ["classify", "detect"], ["detect", "segment"],
    ]

    for label, path, expected in images:
        print(f"\n=== 2. {label} image: {os.path.basename(path)} ===")
        for mods in module_sets:
            r = analyze(path, mods)
            check(f"[{'+'.join(mods)}] HTTP 200", r.status_code == 200, str(r.status_code))
            if r.status_code != 200:
                continue
            j = r.json()
            expected_keys = {RETURN_KEY[m] for m in mods}
            present_keys = {k for k in ("classify", "gradcam", "detect", "segment") if k in j}
            check(f"[{'+'.join(mods)}] returns exactly requested modules",
                  present_keys == expected_keys, f"got {present_keys} want {expected_keys}")
            # annotated images present where expected
            for k in ("gradcam", "detect", "segment"):
                if k in j:
                    check(f"[{'+'.join(mods)}] {k} has image",
                          isinstance(j[k].get("image"), str) and j[k]["image"].startswith("data:image"))
            verdict = ui_verdict(j)
            check(f"[{'+'.join(mods)}] UI verdict == {expected}", verdict == expected,
                  f"got {verdict}")

    print("\n=== 3. Edge cases ===")
    # empty modules -> server default runs all (UI prevents this, but API should be safe)
    r = analyze(fire, [])
    check("empty modules -> HTTP 200 (no crash)", r.status_code == 200, str(r.status_code))

    print(f"\n================  {passed} passed, {failed} failed  ================\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
