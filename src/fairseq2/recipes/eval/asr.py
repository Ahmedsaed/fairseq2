# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, cast

import torch
from datasets import (  # type: ignore[attr-defined,import-untyped,import-not-found]
    Dataset,
    load_dataset,
)

from fairseq2.data.data_pipeline import SequenceData
from fairseq2.data.text import load_text_tokenizer
from fairseq2.datasets.batching import StaticBatching
from fairseq2.datasets.huggingface import Example, create_hf_reader
from fairseq2.logging import get_log_writer
from fairseq2.models import load_model
from fairseq2.models.model import Model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.recipes.eval.configs import (
    HFEvalConfig,
    wav2vec2_presets,
    whisper_presets,
)
from fairseq2.recipes.evaluator import HFEvaluator
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import META, DataType
from fairseq2.utils.profiler import Stopwatch

log = get_log_writer(__name__)


@dataclass
class AsrEvalConfig(HFEvalConfig):
    """Holds the configuration of a ASR evaluation recipe."""

    tokenizer_name: str = "librispeech_asr"
    """The tokenizer to use."""

    split: str = "test"
    """The name of the dataset split to evaluate with."""

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    max_num_elements: int = 3_200_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    max_samples: Optional[int] = None
    """Maximum number of samples from the dataset to be evaluated. Used
    e.g. for debugging. Default is None, meaning all samples will be evaluated"""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    checkpoint_dir: Optional[Path] = None
    """The checkpoint directory containing models saved by a :class:`FileCheckpointManager`."""

    dtype: DataType = torch.float16
    """The data type of the model."""


@wav2vec2_presets.decorator("librispeech_asr")
def _wav2vec2_librispeech_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr",
        model_name="wav2vec2_asr_base_10h",
        split="test.other",
    )


@whisper_presets.decorator("librispeech_asr")
def _whisper_librispeech_asr_config() -> AsrEvalConfig:
    return AsrEvalConfig(
        dataset_name="librispeech_asr", model_name="whisper", split="test.other"
    )


class ASREvaluator:
    def __init__(self) -> None:
        self.gang = setup_root_gang(log)
        self.init_device = self.gang.device if self.gang.rank == 0 else META
        self.wall_watch = Stopwatch(device=self.init_device)

    def to_batch(self, examples: Example) -> Seq2SeqBatch:
        """
        Convert the example data to a batch.

        Args:
            examples: Collated and padded examples.
        """
        source_data = cast(SequenceData, examples["audio"])
        target_data = cast(SequenceData, examples["text"])

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(source_data)
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(target_data)

        return Seq2SeqBatch(
            source_seqs,
            source_padding_mask,
            target_seqs,
            target_padding_mask,
            examples,
        )

    def _preprocess_example(self, example: Example) -> Example:
        """
        Preprocess the example data.

        Note: should be refactored and removed from this class.

        Args:
            example: The example data.

        Returns:
            audio and text tensors.
        """
        audio_tensor = (
            torch.from_numpy(example["audio"]["array"])
            .to(self.config.dtype)
            .to(self.init_device)
        )
        text_tensor = self.encoder(example["text"].lower()).to(self.init_device)
        return {"audio": audio_tensor, "text": text_tensor}

    def preprocessor(self, batch: Seq2SeqBatch) -> Tuple[SequenceBatch, SequenceBatch]:
        """
        Preprocess the batch data.

        Args:
            batch: The batch data.

        Returns:
            A tuple of source and target sequences in the form of SequenceBatch.
        """
        return SequenceBatch(
            batch.source_seqs, batch.source_padding_mask
        ), SequenceBatch(batch.target_seqs, batch.target_padding_mask)

    def postprocesser(
        self, outputs: Any, targets: SequenceBatch
    ) -> Tuple[List[str], List[str]]:
        """
        Postprocess the outputs and targets to get the predictions and references.

        Args:
            outputs: The model outputs.
            targets: The target sequences.
        """
        decoder = self.tokenizer.create_decoder()
        pad_idx = self.tokenizer.vocab_info.pad_idx

        hypotheses, _ = outputs.generate_hypotheses(pad_idx=pad_idx)
        predictions = [decoder(item) for item in hypotheses]
        references = [decoder(item) for item in targets.seqs.to(torch.int32)]

        return predictions, references

    def _load_dataset(
        self, dataset_name: str, split: str, max_samples: Optional[int]
    ) -> Dataset:
        """
        Load a huggingface dataset.

        Args:
            dataset_name: The name of the dataset to load.
            split: The split of the dataset to load.
            max_samples: The maximum number of samples to load.
        """
        iterable_ds = load_dataset(dataset_name, split=split, streaming=True)
        max_samples = cast(int, max_samples if max_samples is not None else math.inf)

        ds = Dataset.from_generator(
            lambda: (
                yield from (
                    item for idx, item in enumerate(iterable_ds) if idx < max_samples
                )
            ),
            features=iterable_ds.features,
        )

        ds = ds.map(self._preprocess_example)
        format = {
            "type": "torch",
            "format_kwargs": {"dtype": self.config.dtype, "device": self.init_device},
        }
        ds.set_format(**format, columns=["audio", "text"])
        return ds

    def _load_model(self, model_name: str) -> Model:
        """
        Load the model.

        Args:
            model_name: The name of the model to load.
        """
        model = load_model(model_name, device=self.init_device, dtype=self.config.dtype)
        return cast(Model, model)

    def _load_evaluator(self) -> HFEvaluator[Seq2SeqBatch]:
        """
        Load the HFEvaluator for ASR.

        Returns:
            The evaluator for ASR.
        """
        pipeline_reader = create_hf_reader(
            dataset=self.dataset,
            gang=self.gang,
            converter=self.to_batch,
            batching=StaticBatching(self.config.max_num_elements),
            num_prefetch=self.config.num_prefetch,
            pad_value=self.tokenizer.vocab_info.pad_idx,
            max_seq_len=self.config.max_audio_len,
        )

        self.wall_watch.start()

        return HFEvaluator[Seq2SeqBatch](
            model=self.model,
            metrics=["bleu"],
            gang=self.gang,
            data_reader=pipeline_reader,
            wall_watch=self.wall_watch,
            preprocessor=self.preprocessor,
            postprocessor=lambda x, y: self.postprocesser(x, y),
        )

    def __call__(self, config: HFEvalConfig, output_dir: Path) -> Callable[[], None]:
        """
        Create an evaluation process for ASR.

        Args:
            config: The configuration of the evaluation process.
            output_dir: The directory to store the evaluation results.

        Returns:
            A callable that will run the evaluation process
        """
        if not isinstance(config, AsrEvalConfig):
            raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

        self.config = config
        self.output_dir = output_dir

        self.model = self._load_model(config.model_name)
        self.tokenizer = load_text_tokenizer(config.tokenizer_name)
        self.encoder = self.tokenizer.create_encoder(device=self.init_device)
        self.dataset = self._load_dataset(
            config.dataset_name, config.split, config.max_samples
        )

        return self._load_evaluator()
