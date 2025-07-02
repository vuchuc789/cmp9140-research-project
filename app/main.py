import argparse

from app.data.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="(re)preprocess raw data",
    )

    args = parser.parse_args()

    if args.preprocess:
        preprocess()

    parser.print_help()


if __name__ == "__main__":
    main()
