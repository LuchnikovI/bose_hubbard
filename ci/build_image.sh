#!/usr/bin/env bash

ci_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${ci_dir}/utils.sh"

project_dir="$(dirname ${ci_dir})"

# ------------------------------------------------------------------------------------------

log INFO "Building an image..."
cat > "${BH_IMAGE_NAME}.def" <<EOF

Bootstrap: docker
From: ${BH_RUST_IMAGE}
Stage: build
%files
    "${project_dir}/.cargo" /bose_hubbard/.cargo
    "${project_dir}/src" /bose_hubbard/src
    "${project_dir}/Cargo.toml" /bose_hubbard/Cargo.toml
    "${project_dir}/Cargo.lock" /bose_hubbard/Cargo.lock
%post
    apt-get update && apt-get install -y python3-pip python3-venv
    python3 -m pip install --break-system-packages --no-cache --upgrade \
        pip \
        setuptools \
        maturin \
        patchelf \
        numpy
    python3 -m maturin build --release --manifest-path /bose_hubbard/Cargo.toml
    (cd /bose_hubbard && cargo t)

Bootstrap: docker
From: ${BH_BASE_IMAGE}
Stage: final
%files from build
    /bose_hubbard/target/wheels /bose_hubbard/wheels
%setup
    chmod +x "$(dirname ${ci_dir})/entrypoint.sh"
    chmod +x "$(dirname ${ci_dir})/pysrc/run_dynamics_cli.py"
    chmod +x "$(dirname ${ci_dir})/pysrc/plotting_cli.py"
%post
    apt-get update && apt-get install -y python3-pip
    python3 -m pip install --break-system-packages --no-cache-dir --upgrade \
        pip \
        setuptools \
        numpy==1.25.1 \
        scipy==1.11.1 \
        matplotlib==3.7.2 \
        pytest==7.4.0 \
        pyyaml==6.0 \
        mypy==1.4.1 \
        h5py==3.9.0
    for wheel in /bose_hubbard/wheels/*
    do
        python3 -m pip install --break-system-packages "\${wheel}"
    done

%runscript
    "$(dirname "${ci_dir}")/entrypoint.sh \$@"
EOF

# ------------------------------------------------------------------------------------------

if singularity build -F "${BH_IMAGE_NAME}.sif" "${BH_IMAGE_NAME}.def";
then
    log INFO "Base image ${BH_IMAGE_NAME} has been built"
    rm -f "${BH_IMAGE_NAME}.def"
else
    log ERROR "Failed to build a base image ${BH_IMAGE_NAME}"
    rm -f "${BH_IMAGE_NAME}.def"
    exit 1
fi