from fairseq2.recipes.cli import Cli
from fairseq2.recipes.intro.intro import IntroCommandHandler


def _setup_intro_cli(cli: Cli) -> None:
    group = cli.add_group("intro", help="Introduction to the CLI")

    group.add_command(
        name="display",
        handler=IntroCommandHandler(),
        help="Display a hello world message",
    )
