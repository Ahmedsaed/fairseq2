from fairseq2.recipes.cli import Cli, CliCommandHandler
from argparse import ArgumentParser, Namespace
from fairseq2.typing import override


class IntroCommandHandler(CliCommandHandler):
    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        """Initialize parser with command-specific arguments."""
        # No arguments needed for this simple command

    @override
    def __call__(self, args: Namespace) -> None:
        """Run the command."""
        print("Hello World")
