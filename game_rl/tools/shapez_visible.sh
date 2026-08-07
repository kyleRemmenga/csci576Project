#!/usr/bin/env bash
# Launch shapez with a VISIBLE window while keeping the RL API usable.
#
# The RL API only answers when the page carries ?rl-headless=1, but that same
# flag normally disables core.draw(), so an RL-controlled game is invisible.
# tools/patch_rl_render.py adds an independent rl-render=1 flag that re-enables
# drawing. Electron itself is launched WITHOUT --rl-headless so the window shows.
#
# Copy into WSL and run:
#   wsl -d Ubuntu -- cp /mnt/c/.../game_rl/tools/shapez_visible.sh /home/fahad/
#   wsl -d Ubuntu -- bash /home/fahad/shapez_visible.sh
set -euo pipefail

web_port="${SHAPEZ_WEB_PORT:-3005}"
rl_port="${SHAPEZ_RL_API_PORT:-17872}"
repo="${SHAPEZ_REPO:-/home/fahad/shapez.io_rl}"
web_root="$(readlink -f "$repo/result")/share/shapez.io"
app_dir="$(ls -d /nix/store/*shapez.io-electron-app-*/app | head -1)"
electron="$(ls -d /nix/store/*electron-16.2.8/bin/electron | head -1)"
python3bin="$(ls -d /nix/store/*python3-3.10.13/bin/python3 | head -1)"

"$python3bin" -m http.server "$web_port" --bind 127.0.0.1 --directory "$web_root" >"/tmp/shapez-web-$web_port.log" 2>&1 &
web_pid=$!
trap 'kill "$web_pid" >/dev/null 2>&1 || true' EXIT INT TERM

for _ in $(seq 1 50); do
  curl -fsS "http://127.0.0.1:$web_port/" >/dev/null 2>&1 && break
  sleep 0.1
done
echo "web:  http://127.0.0.1:$web_port"
echo "rl:   http://127.0.0.1:$rl_port"

export SHAPEZ_RL_API=1
export SHAPEZ_RL_API_PORT="$rl_port"
# Chromium only parses --switch=value, so the URL must also come through the env
# var the app falls back to. rl-headless=1 keeps the RL API responding;
# rl-render=1 re-enables drawing (see tools/patch_rl_render.py).
export SHAPEZ_LOCAL_URL="http://127.0.0.1:$web_port/?rl-headless=1&rl-render=1"
unset SHAPEZ_RL_HEADLESS ELECTRON_RUN_AS_NODE || true

# WSLg has no GLX (Chromium's default GL path), but it does expose a real GPU
# through EGL/Mesa-D3D12. Try that first for smoother FPS; SHAPEZ_GL=swiftshader
# falls back to pure software rendering if EGL fails to init.
gl_backend="${SHAPEZ_GL:-egl}"
exec "$electron" "$app_dir" \
  --local \
  "--local-url=$SHAPEZ_LOCAL_URL" \
  "--rl-user-data-dir=/tmp/shapez-rl-visible-$rl_port" \
  --use-gl="$gl_backend" \
  --enable-unsafe-swiftshader
