from argparse import ArgumentParser
import sys


def parse_args(args=sys.argv[1:]):
    parser = ArgumentParser()

    # training config
    parser.add_argument("--model-str", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--dataset-str", type=str, default="esnli")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # explanations
    parser.add_argument("--explanations", type=str, nargs="*", default=[])
    parser.add_argument("--explanation-model", type=str, default="gpt4o")
    parser.add_argument("--explanation-type", type=str, default="ZS")
    parser.add_argument("--method", type=str, default="append")  # "prepend"
    parser.add_argument("--explanation-targets", type=str, nargs="*", default=[])

    # logging
    parser.add_argument("--project", type=str)

    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--save-model", action="store_true")

    args = parser.parse_args(args)
    return args
