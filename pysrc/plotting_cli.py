#!/usr/bin/python3

import argparse
from argparse import RawTextHelpFormatter
import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np

description = "This app plots results of a Bose-Hubbard system simulation."
result_help = "a path to a directory with results."

def main():
    parser = argparse.ArgumentParser(
        prog='plotting',
        description=description,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--result", "-r", help=result_help)
    args = parser.parse_args()
    log_file = f"{args.result}/log.yaml"
    data_file = f"{args.result}/data.hdf5"
    with open(log_file, 'r') as f:
        config = yaml.safe_load(f)
    with h5py.File(data_file, 'r') as data:
        for statistic in config["config"]["diagonal_statistics"]:
            match statistic:
                case "n_bosons_weight":
                    plt.figure()
                    plt.plot(np.array(data[statistic]))
                    plt.legend([str(i) for i in range(data[statistic].shape[0])])
                    plt.savefig(f"{args.result}/n_bosons_weight.pdf")
                    plt.figure()
                    plt.plot(np.array(data[statistic]))
                    plt.legend([str(i) for i in range(data[statistic].shape[0])])
                    plt.yscale('log')
                    plt.savefig(f"{args.result}/log_n_bosons_weight.pdf")
                case "particles_per_mode":
                    plt.figure()
                    plt.plot(data[statistic])
                    plt.savefig(f"{args.result}/particles_per_mode.pdf")
                    plt.figure()
                    plt.imshow(data[statistic])
                    plt.savefig(f"{args.result}/particles_per_mode_image.pdf")

if __name__ == '__main__':
    main()