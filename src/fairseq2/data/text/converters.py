# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE
from torch import Tensor

from fairseq2.typing import DataType

if TYPE_CHECKING or DOC_MODE:

    @final
    class StrSplitter:
        """Split string on a given character.

        :param sep:
            The character to split on (default to tab)

        :param names:
            names of the corresponding columns of the input tsv file
            Will create dictionaries object with one entry per column

        :param indices:
            The indices of the column to keep.

        Example usage::

            # read all columns: ["Go.", "Va !", "CC-BY 2.0 (France)"]
            dataloader = read_text("tatoeba.tsv").map(StrSplitter()).and_return()
            # keep only the second column and convert to string: "Va !"
            dataloader = read_text("tatoeba.tsv").map(StrSplitter(indices=[1])).map(lambda x: x[0]).and_return()
            # keep only the first and second column and convert to dict: {"en": "Go.", "fr": "Va !"}
            dataloader = read_text("tatoeba.tsv").map(StrSplitter(names=["en", "fr"], indices=[0, 1])).and_return()

        """

        def __init__(
            self,
            sep: str = "\t",
            names: Sequence[str] | None = None,
            indices: Sequence[int] | None = None,
            exclude: bool = False,
        ) -> None:
            ...

        def __call__(self, s: str) -> list[str] | dict[str, str]:
            ...

    @final
    class StrToIntConverter:
        """Parses integers in a given base"""

        def __init__(self, base: int = 10) -> None:
            ...

        def __call__(self, s: str) -> int:
            ...

    @final
    class StrToTensorConverter:
        def __init__(
            self,
            size: Sequence[int] | None = None,
            dtype: DataType | None = None,
        ) -> None:
            ...

        def __call__(self, s: str) -> Tensor:
            ...

else:
    from fairseq2n.bindings.data.text.converters import StrSplitter as StrSplitter
    from fairseq2n.bindings.data.text.converters import (
        StrToIntConverter as StrToIntConverter,
    )
    from fairseq2n.bindings.data.text.converters import (
        StrToTensorConverter as StrToTensorConverter,
    )

    def _set_module_name() -> None:
        for t in [StrSplitter, StrToIntConverter, StrToTensorConverter]:
            t.__module__ = __name__

    _set_module_name()
