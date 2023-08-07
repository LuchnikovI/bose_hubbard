#!/usr/bin/python3

import argparse
from argparse import RawTextHelpFormatter
from utils import Config

description = "This app simulates exact dynamics of the Bose-Hubbard model."
config_help = "a path to a *.yaml config file with Bose-Hubbard model parameters"
result_help = "a path to the directory with simulation results"

def main():
    parser = argparse.ArgumentParser(
        prog='run_dynamics',
        description=description,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", help=config_help)
    parser.add_argument("--result", "-r", help=result_help)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config(f.read())
    config.run_bose_hubbard_dynamics(args.result)




if __name__ == '__main__':
    main()