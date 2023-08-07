#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
entrypoint="${script_dir}/entrypoint.sh"

. "${script_dir}/ci/utils.sh"

. "${script_dir}/ci/ensure_image.sh"

singularity exec --cleanenv "${BH_IMAGE_NAME}.sif" "${entrypoint}" "$@"