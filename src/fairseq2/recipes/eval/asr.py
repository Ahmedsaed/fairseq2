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
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2.asr import load_wav2vec2_asr_model
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

    def _librispeech_asr_to_batch(self, examples: Example) -> Seq2SeqBatch:
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
        audio_tensor = (
            torch.from_numpy(example["audio"]["array"])
            .to(torch.float16)
            .to(self.init_device)
        )
        text_tensor = self.encoder(example["text"].lower()).to(self.init_device)
        return {"audio": audio_tensor, "text": text_tensor}

    def seq2seq_preprocessor(
        self, batch: Seq2SeqBatch
    ) -> Tuple[SequenceBatch, SequenceBatch]:
        return SequenceBatch(
            batch.source_seqs, batch.source_padding_mask
        ), SequenceBatch(batch.target_seqs, batch.target_padding_mask)

    def postprocesser(
        self, outputs: Any, targets: SequenceBatch
    ) -> Tuple[List[str], List[str]]:
        decoder = self.tokenizer.create_decoder()
        pad_idx = self.tokenizer.vocab_info.pad_idx

        hypotheses, _ = outputs.generate_hypotheses(pad_idx=pad_idx)
        predictions = [decoder(item) for item in hypotheses]
        references = [decoder(item) for item in targets.seqs.to(torch.int32)]

        return predictions, references

    def _load_evaluator(self) -> HFEvaluator[Seq2SeqBatch]:
        iterable_ds = load_dataset(
            self.config.dataset_name, split=self.config.split, streaming=True
        )
        max_samples = (
            self.config.max_samples if self.config.max_samples is not None else math.inf
        )

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
            "format_kwargs": {"dtype": torch.float16, "device": self.init_device},
        }
        ds.set_format(**format, columns=["audio", "text"])

        pipeline_reader = create_hf_reader(
            dataset=ds,
            gang=self.gang,
            converter=self._librispeech_asr_to_batch,
            batching=StaticBatching(self.config.max_num_elements),
            num_prefetch=self.config.num_prefetch,
            pad_value=self.tokenizer.vocab_info.pad_idx,
            max_seq_len=self.config.max_audio_len,
        )

        model = load_wav2vec2_asr_model(
            self.config.model_name, device=self.init_device, dtype=self.config.dtype
        )

        wall_watch = Stopwatch(start=True, device=self.init_device)

        return HFEvaluator[Seq2SeqBatch](
            model=model,
            metrics=["bleu"],
            gang=self.gang,
            data_reader=pipeline_reader,
            wall_watch=wall_watch,
            preprocessor=self.seq2seq_preprocessor,
            postprocessor=lambda x, y: self.postprocesser(x, y),
        )

    def __call__(self, config: HFEvalConfig, output_dir: Path) -> Callable[[], None]:
        """
        This method will run the evaluation process.

        Returns:
            A callable that will run the evaluation process
        """
        if not isinstance(config, AsrEvalConfig):
            raise ValueError(f"Expect AsrEvalConfig, get {type(config)}")

        self.config = config
        self.output_dir = output_dir

        self.tokenizer = load_text_tokenizer(config.tokenizer_name)
        self.encoder = self.tokenizer.create_encoder(device=self.init_device)

        return self._load_evaluator()
