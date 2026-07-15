#!/usr/bin/env bash
# Deploy Homie to the Raspberry Pi. Two trees, two systemd services:
#   pi/        → the voice app          → homie.service
#   smarthome/ → the dashboard + kiosk  → smarthome.service
#
# The Pi cannot reliably resolve github.com, so deploys are Mac-driven rather than
# a git pull on the Pi: export the committed trees and rsync them over Tailscale
# SSH, then reinstall deps and restart the systemd services. The Pi's runtime state
# is preserved — .env, .venv, .homie-state, and secrets/ (one level up) are never
# touched. smarthome/ syncs WITHOUT --delete so untracked device state on the Pi
# (.electra_imei, midea_devices.json) survives.
#
# Deploys committed HEAD, not the working tree — so uncommitted WIP never reaches
# the live device. Commit what you want live first.
#
# Usage:
#   ./deploy-pi.sh          deploy HEAD + restart the service
#   ./deploy-pi.sh -n       dry run: print what would change, touch nothing
#
# Target overrides:
#   HOMIE_PI_HOST=akiva@100.111.144.6 HOMIE_PI_DIR=heyhomie ./deploy-pi.sh

set -euo pipefail

PI_HOST="${HOMIE_PI_HOST:-akiva@100.111.144.6}"
PI_DIR="${HOMIE_PI_DIR:-heyhomie}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS=(-o ConnectTimeout=20)

DRY=()
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
  DRY=(--dry-run)
  echo "▶ DRY RUN — no changes will be made"
fi

# Export committed HEAD to a temp tree so only tracked files ship (no WIP, no junk).
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$REPO_ROOT" archive HEAD pi smarthome | tar -x -C "$WORK"
echo "▶ Deploying $(git -C "$REPO_ROOT" rev-parse --short HEAD) ($(git -C "$REPO_ROOT" log -1 --format=%s))"

# Anchored excludes (/foo) protect the Pi's runtime dirs from --delete.
EXCLUDES=(
  --exclude='/.env' --exclude='/.venv' --exclude='/venv' --exclude='/.homie-state'
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache'
  --exclude='.claude' --exclude='.DS_Store' --exclude='*.wav' --exclude='*.aiff'
)

echo "▶ Syncing pi/ → ${PI_HOST}:${PI_DIR}/pi/"
rsync -rlptz --delete -i ${DRY[@]+"${DRY[@]}"} "${EXCLUDES[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${WORK}/pi/" "${PI_HOST}:${PI_DIR}/pi/"

# No --delete here: the Pi keeps untracked device state alongside the code.
echo "▶ Syncing smarthome/ → ${PI_HOST}:${PI_DIR}/smarthome/"
rsync -rlptz -i ${DRY[@]+"${DRY[@]}"} "${EXCLUDES[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "${WORK}/smarthome/" "${PI_HOST}:${PI_DIR}/smarthome/"

if ((${#DRY[@]})); then
  echo "✓ Dry run complete. Re-run without -n to apply."
  exit 0
fi

echo "▶ Deps + compile check + service reload + restart…"
ssh "${SSH_OPTS[@]}" "${PI_HOST}" "set -e
  cd ${PI_DIR}/pi
  .venv/bin/pip install -q -r requirements.txt
  # Fail before touching the running service if the synced tree won't compile.
  .venv/bin/python -m compileall -q homie
  if ! cmp -s homie.service /etc/systemd/system/homie.service; then
    echo '  systemd unit changed → updating + daemon-reload'
    sudo cp homie.service /etc/systemd/system/homie.service
    sudo systemctl daemon-reload
  fi
  sudo systemctl restart homie
  sleep 3
  echo \"  status: \$(systemctl is-active homie)\"

  cd ${PI_DIR}/smarthome
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/python -m compileall -q server.py higoal_cli.py electra_cli.py midea_cli.py higoal_client
  sudo systemctl restart smarthome
  sleep 2
  echo \"  smarthome: \$(systemctl is-active smarthome)\""

echo "▶ Recent logs:"
ssh "${SSH_OPTS[@]}" "${PI_HOST}" "journalctl -u homie -u smarthome -n 15 --no-pager -o cat"
echo "✓ Deployed to ${PI_HOST}."
