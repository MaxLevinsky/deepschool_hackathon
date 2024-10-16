import argparse

from pruner import run_pruner


def run_pr():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset_yaml_path", default="/pruner/dataset_yaml", help="")
    parser.add_argument("-m", "--model_path", default="/pruner/models", help="")
    parser.add_argument("-o", "--output", default="/pruner/output", help="")

    args = parser.parse_args()

    run_pruner(dataset_yaml_path=args.dataset_yaml_path, model_path=args.model_path, output_path=args.output)