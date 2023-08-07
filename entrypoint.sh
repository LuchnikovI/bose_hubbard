#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

get_help() {
cat << EOF
This is an utility for exact numerial simulation of
Bose-Hubbard model's dynamics.

Usage: run.sh COMMAND

COMMANDs:
  test:         runs tests (pytest, rust part is tested during build);
  typecheck:    runs static code analysis for python;
  run_dynamics: runs dynamics simulation (use run.sh run_dynamics --help for more information)
  plot:         plots results of numerical experiments (use run.sh plot --help for more information)
  --help:       drops this message;
EOF
}

case $1 in

  test)
        python3 -m pytest "${script_dir}/pysrc"
    ;;
  typecheck)
        python3 -m mypy "${script_dir}/pysrc"
    ;;
  --help | -h)
        get_help
    ;;
  run_dynamics)
        shift
        "${script_dir}/pysrc/run_dynamics_cli.py" "$@"
  ;;
  plot)
        shift
        "${script_dir}/pysrc/plotting_cli.py" "$@"
  ;;
  *)
        echo "Unknown option: '$1'"
        get_help
        exit 1
    ;;

esac