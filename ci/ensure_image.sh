#!/usr/bin/env bash

ci_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${ci_dir}/utils.sh"

# ------------------------------------------------------------------------------------------


if [[ -f "${BH_IMAGE_NAME}.sif" ]]; then
    :
else
    log INFO "${BH_IMAGE_NAME}.sif image has not been found"
    . "${ci_dir}/build_image.sh"
fi