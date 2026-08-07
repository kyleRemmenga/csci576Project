#!/usr/bin/env python3
"""Patch the shapez fork so a headless RL session can still render to a window.

The RL API only responds when `app.rlHeadless` is true, but that same flag
disables `core.draw()`, so an RL-controlled game is normally invisible. This adds
an independent `rl-render=1` query flag that re-enables drawing without changing
any RL behaviour, so an episode can be screen-recorded.

    python tools/patch_rl_render.py /home/fahad/shapez.io_rl
"""

import argparse
import pathlib
import sys

ENDPOINT_OLD = """    app.rlHeadless = new URLSearchParams(window.location.search).get("rl-headless") === "1";"""
ENDPOINT_NEW = """    app.rlHeadless = new URLSearchParams(window.location.search).get("rl-headless") === "1";
    // Lets a headless RL session still draw, so episodes can be screen-recorded.
    app.rlRender = new URLSearchParams(window.location.search).get("rl-render") === "1";"""

INGAME_OLD = """            if (this.app.pageVisible && !isHeadless) {
                this.core.draw();
            }"""
INGAME_NEW = """            if (this.app.pageVisible && (!isHeadless || this.app.rlRender === true)) {
                this.core.draw();
            }"""


def patch(path, old, new, label):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  {label}: already patched")
        return False
    if text.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {text.count(old)} in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"  {label}: patched")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Path to the shapez.io_rl checkout")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo)
    endpoint = repo / "src/js/rl/rl_endpoint.js"
    ingame = repo / "src/js/states/ingame.js"
    for target in (endpoint, ingame):
        if not target.is_file():
            raise SystemExit(f"missing {target}")

    changed = patch(endpoint, ENDPOINT_OLD, ENDPOINT_NEW, "rl_endpoint.js")
    changed |= patch(ingame, INGAME_OLD, INGAME_NEW, "ingame.js")
    print("rebuild required" if changed else "no changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
