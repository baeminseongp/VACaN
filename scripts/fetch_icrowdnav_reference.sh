#!/usr/bin/env bash
set -euo pipefail

destination="${1:-external/icrowdnav}"
commit="5dc3eda6db8b0e23f790eb215aae8de27e07b744"

if [[ -e "$destination" ]]; then
  echo "Refusing to overwrite existing path: $destination" >&2
  exit 2
fi

git clone https://github.com/BRoln7/icrowdnav.git "$destination"
git -C "$destination" checkout --detach "$commit"
echo "Fetched iCrowdNav $commit into $destination"
