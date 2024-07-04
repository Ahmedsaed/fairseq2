from typing import List
from fairseq2.recipes.cli import Cli, CliCommandHandler
from argparse import ArgumentParser, Namespace
from fairseq2.typing import override
from datasets import load_dataset_builder, load_dataset
from evaluate import load as load_metric
from tabulate import tabulate


class DatasetCommandHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize parser with command-specific arguments."""
        parser.add_argument("name", help="name of the metric")

        parser.add_argument(
            "-c",
            "--column",
            type=str,
            default="",
            dest="column",
            nargs="*",
            help="Column names to explore",
        )

        parser.add_argument(
            "-t",
            "--top",
            type=int,
            default=10,
            dest="top",
            help="Number of top rows to show",
        )

    @override
    def __call__(self, args: Namespace) -> None:
        """
        :param args:
            The parsed command-line arguments.
        """
        info = self.get_dataset_info(args.name)
        self.print_dataset_info(info)

        dataset_loader = self.load_dataset(args.name)
        self.print_dataset(dataset_loader, top=args.top, columns=args.column)

    def get_dataset_info(self, name: str):
        """
        :param name:
            The name of the dataset to explore.
        :return:
            The dataset info.
        """
        builder = load_dataset_builder(name)
        return builder.info

    def load_dataset(self, name: str):
        """
        :param name:
            The name of the dataset to explore.
        :return:
            The dataset loader.
        """
        return load_dataset(name, streaming=True)

    def print_dataset_info(self, dataset_info):
        basic_info = [
            ["Dataset Name", dataset_info.dataset_name],
            ["Builder Name", dataset_info.builder_name],
            ["Configuration Name", dataset_info.config_name],
            ["Version", dataset_info.version]
        ]

        if dataset_info.description:
            basic_info.append(["Description", dataset_info.description])
        if dataset_info.homepage:
            basic_info.append(["Homepage", dataset_info.homepage])
        if dataset_info.license:
            basic_info.append(["License", dataset_info.license])

        print("Dataset Information:")
        print(tabulate(basic_info, headers=[
              "Field", "Value"], tablefmt="pretty"))
        print()

    def print_dataset(self, dataset_loader, top: int = 10, columns: List[str] = []):
        """
        :param dataset_loader:
            The dataset loader.
        :param top:
            Number of top rows to show.
        :param columns:
            Column names to explore.
        """
        if not columns:
            # If no columns are specified, use all columns from the first row
            first_row = next(iter(dataset_loader['train']))
            columns = list(first_row.keys())

        data = []
        for i, row in enumerate(dataset_loader['train']):
            if i >= top:
                break
            data.append([str(row[col])[:70] for col in columns])

        print("Dataset Records:")
        print(tabulate(data, headers=columns, tablefmt="pretty"))


class MetricCommandHandler(CliCommandHandler):
    @ override
    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize parser with command-specific arguments."""
        parser.add_argument("name", help="name of the metric")

    @ override
    def __call__(self, args: Namespace) -> None:
        """
        :param args:
            The parsed command-line arguments.
        """
        metric = load_metric(args.name)
        self.print_metric_info(metric)

    def print_metric_info(self, metric):
        print(f"Metric Name: {metric.name}")
        print(f"Metric Description: {metric.description}")
