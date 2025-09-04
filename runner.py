from custom_function_boost_search_v2 import train_run
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
    args = parser.parse_args()
    train_run(args.v1)
