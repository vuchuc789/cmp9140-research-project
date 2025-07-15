import argparse

from app.data.analyze import analyze
from app.data.downsample import downsample
from app.data.partition import view_distribution
from app.data.preprocess import preprocess
from app.model.evaluate import evaluate
from app.model.train import fit_model, init_model


def main():
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="CMP9140 Research Project - DDoS Detection with Federated Learning",
        epilog="Author: Chuc Van Vu (29630583@students.lincoln.ac.uk)",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()

    data_parser = subparsers.add_parser("data", help="process data")
    data_parser.add_argument(
        "--preprocess",
        action="store_true",
        help="(re)preprocess raw data",
    )
    data_parser.add_argument(
        "--downsample",
        action="store_true",
        help="downsample huge subsets",
    )
    data_parser.add_argument(
        "--anomaly-ratio",
        default="0.2",
        help="anomaly ratio",
    )
    data_parser.add_argument(
        "--analyze",
        action="store_true",
        help="show data analysis",
    )
    data_parser.add_argument(
        "--analyze-file",
        default="data/Benign.parquet.zst",
        help="path to data file",
    )
    data_parser.add_argument(
        "--partition",
        help="visualize partition distributions",
    )
    data_parser.set_defaults(func=lambda args: data_command(data_parser, args))

    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument(
        "--centralized",
        action="store_true",
        help="proceed centralized training",
    )
    train_parser.add_argument(
        "--model-name",
        default="centralized",
        help="model name",
    )
    train_parser.add_argument(
        "--epochs",
        default="5",
        help="number of epochs",
    )
    train_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate model",
    )
    train_parser.set_defaults(func=lambda args: train_command(train_parser, args))

    args = parser.parse_args()
    args.func(args)


def data_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    print_help = True

    if args.preprocess:
        print_help = False
        preprocess()

    if args.downsample:
        print_help = False
        downsample(float(args.anomaly_ratio))

    if args.analyze:
        print_help = False
        analyze(args.analyze_file)

    if args.partition:
        print_help = False
        view_distribution(args.partition)

    if print_help:
        parser.print_help()


def train_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    print_help = True

    if args.centralized:
        print_help = False
        config = init_model(model_name=args.model_name, verbose=True)
        fit_model(
            *config,
            epochs=int(args.epochs),
            model_name=args.model_name,
            verbose=True,
        )

    if args.evaluate:
        print_help = False
        evaluate(args.model_name)

    if print_help:
        parser.print_help()


if __name__ == "__main__":
    main()
