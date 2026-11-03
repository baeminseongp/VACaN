#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference_root="${ICROWDNAV_ROOT:-$project_root/external/icrowdnav}"
isaac_root="${ISAACSIM_ROOT:-}"

if [[ -z "$isaac_root" || ! -x "$isaac_root/python.sh" ]]; then
  echo "Set ISAACSIM_ROOT to an Isaac Sim 4.2.0 installation containing python.sh." >&2
  exit 2
fi
if [[ ! -d "$reference_root/icrowdnav/src/scene/PegasusSimulator/extensions/pegasus.simulator" ]]; then
  echo "Missing iCrowdNav reference. Run: bash scripts/fetch_icrowdnav_reference.sh external/icrowdnav" >&2
  exit 2
fi

pegasus_extension="$reference_root/icrowdnav/src/scene/PegasusSimulator/extensions/pegasus.simulator"
export PYTHONPATH="$pegasus_extension:${PYTHONPATH:-}"

exec "$isaac_root/python.sh" \
  "$project_root/scripts/isaacsim/run_pegasus_environment.py" \
  --config "$project_root/configs/isaacsim/pegasus_warehouse.yaml" \
  "$@"
