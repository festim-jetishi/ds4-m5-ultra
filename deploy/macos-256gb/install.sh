#!/usr/bin/env bash
set -euo pipefail

# Install persistent 236 GiB Metal wired-memory limit + 1M x 6 ds4-server LaunchAgent
# for Apple Mac Studio M5 Ultra (256 GiB Unified Memory).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS4_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USER_UID="$(id -u)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: this installer is for macOS (Apple Silicon) only." >&2
  exit 1
fi

MEM_BYTES="$(sysctl -n hw.memsize)"
MEM_GIB="$((MEM_BYTES / 1024 / 1024 / 1024))"
if (( MEM_GIB < 256 )); then
  echo "Error: detected ${MEM_GIB} GiB RAM; iogpu.wired_limit_mb=241664 requires >= 256 GiB." >&2
  echo "Adjust wired_limit_mb and --ctx / --batched-session manually for smaller Macs." >&2
  exit 1
fi

if [[ ! -x "${DS4_DIR}/ds4-server" ]]; then
  echo "Building ds4-server in ${DS4_DIR}..."
  make -C "${DS4_DIR}" -j8 ds4-server
fi

echo "1/2 Installing LaunchDaemon (iogpu.wired_limit_mb=241664 + disable_wired_collector=1)..."
sudo cp "${SCRIPT_DIR}/local.ds4.iogpu-wired-limit.plist" /Library/LaunchDaemons/local.ds4.iogpu-wired-limit.plist
sudo chown root:wheel /Library/LaunchDaemons/local.ds4.iogpu-wired-limit.plist
sudo chmod 644 /Library/LaunchDaemons/local.ds4.iogpu-wired-limit.plist
sudo sysctl -w iogpu.wired_limit_mb=241664 iogpu.disable_wired_collector=1
sudo launchctl bootout system/local.ds4.iogpu-wired-limit 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/local.ds4.iogpu-wired-limit.plist

echo "2/2 Installing LaunchAgent (1,048,576 x 6 slots, DSpark exact sampling, A44 disk KV)..."
mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/.ds4/server-kv"
AGENT_DST="${HOME}/Library/LaunchAgents/local.ds4.server.plist"
sed -e "s|__DS4_DIR__|${DS4_DIR}|g" -e "s|__HOME__|${HOME}|g" \
  "${SCRIPT_DIR}/local.ds4.server.plist" > "${AGENT_DST}"

launchctl bootout "gui/${USER_UID}/local.ds4.server" 2>/dev/null || true
launchctl bootstrap "gui/${USER_UID}" "${AGENT_DST}"

echo "Installed! Watch startup logs with:"
echo "  tail -f ${HOME}/ds4-server.log"
