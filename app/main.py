import argparse
import sys

from app.data.analyze import analyze
from app.data.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="CMP9140 Research Project - DDoS Detection with Federated Learning",
        epilog="Author: Chuc Van Vu (29630583@students.lincoln.ac.uk)",
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="(re)preprocess raw data",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="show data analysis",
    )

    args = parser.parse_args()

    if args.preprocess:
        preprocess()

    if args.analyze:
        analyze()

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    main()
