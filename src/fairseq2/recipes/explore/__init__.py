from fairseq2.recipes.cli import Cli
from fairseq2.recipes.explore.explore import DatasetCommandHandler, MetricCommandHandler


def _setup_explore_cli(cli: Cli) -> None:
    group = cli.add_group(
        "explore", help="Explore HuggingFace Datasets And Metrics")

    group.add_command(
        name="dataset",
        handler=DatasetCommandHandler(),
        help="Explore a HuggingFace Dataset",
    )

    group.add_command(
        name="metric",
        handler=MetricCommandHandler(),
        help="Explore a HuggingFace Metric",
    )
