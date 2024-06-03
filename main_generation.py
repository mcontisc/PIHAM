import numpy as np
from argparse import ArgumentParser
from src import synthetic


def main_generate_data():
    p = ArgumentParser()
    p.add_argument(
        "-n", "--samples", type=int, default=1
    )  # number of independent samples
    p.add_argument("-r", "--rseed", type=int, default=0)  # random seed
    p.add_argument("-K", "--K", type=int, default=3)  # number of communities
    p.add_argument("-N", "--N", type=int, default=200)  # number of nodes
    args = p.parse_args()

    rseed = args.rseed
    np.random.seed(rseed)
    L = 3  # number of layers

    for _ in range(args.samples):
        rseed += np.random.randint(1, args.samples+1)
        synthetic.StandardPIHAM(K=args.K, N=args.N, L=L, seed=rseed)


if __name__ == "__main__":
    main_generate_data()
