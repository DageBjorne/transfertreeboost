from experiments_src.simulations import train_run
from experiments_src.tester import train_run_tester
from experiments_src.simulations_trada import train_run_trada
import argparse

#v1_list = [0.005, 0.007, 0.01, 0.03, 0.07]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument(
        "--v1",
        dest="v1",
        default="0",
        help="idx of val of v1",
    )
    parser.add_argument(
        "--ver",
        dest="v2",
        default="sim",
        help="idx of val of v1",
    )
    args = parser.parse_args()
    if args.v2 == 'sim':
        train_run(args.v1)
    elif args.v2 == 'rs':
        train_run_tester(args.v1)
    else:
        train_run_trada(args.v1)
