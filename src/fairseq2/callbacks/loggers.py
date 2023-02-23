import dataclasses
import logging
import os
import typing as tp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import torch
import torchtnt.utils.distributed
import torchtnt.utils.loggers
import torchtnt.utils.timer
import yaml
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TTrainUnit
from torchtnt.utils.loggers import Scalar

log = logging.getLogger(__name__)

# TODO: This should be a helper somewhere
class Stateful:
    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        return self.state_dict()

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        return self.load_state_dict(state)

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)


class MetricLogger(Stateful):
    def __init__(self, config_file: Path):
        assert config_file.parent.is_dir()
        self.config_file = config_file

    def read_config(self) -> Any:
        if not self.config_file.exists():
            return {}
        return yaml.load(self.config_file.read_text(), Loader=yaml.Loader)

    def prepare(self) -> None:
        pass

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        ...

    def close(self) -> None:
        pass


def collect_job_info() -> Dict[str, str]:
    job = {}
    if "SLURM_JOB_ID" in os.environ:
        job["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if "HOST" in os.environ:
        job["host"] = os.environ["HOST"]
    return job


class StdoutLogger(MetricLogger):
    # Should loggers be responsible for updating the config file ?
    def prepare(self) -> None:
        config = yaml.load(self.config_file.read_text(), Loader=yaml.Loader)
        config["job"] = collect_job_info()
        self.config_file.write_text(yaml.dump(config))
        print(config)

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        print("Step:", step, payload)


class WandbLogger(MetricLogger):
    def __init__(
        self,
        config_file: Path,
        project: str,
        job_type: str,
        group_id: str,
    ):
        import wandb

        super().__init__(config_file)
        self.project = project
        self.group_id = group_id
        self.job_type = job_type
        self.run_id: tp.Optional[str] = None
        self._rank: int = torchtnt.utils.distributed.get_global_rank()
        self._wandb = wandb
        self._wandb_run: tp.Any = None

    def prepare(self) -> None:
        if self._wandb_run is not None:
            return
        if self._rank != 0:
            return

        wandb = self._wandb
        if self.run_id is not None:
            # Resume existing run
            entity, project, run_id = self.run_id.split("/")
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                id=run_id,
                resume="must",
            )
            return

        if "/" in self.project:
            entity, project = self.project.split("/", 1)
        else:
            entity, project = None, self.project  # type: ignore

        config = self.read_config()
        run = wandb.init(
            project=project,
            entity=entity,
            config=_simple_conf(config),
            group=self.group_id,
            job_type=self.job_type,
        )
        if run is None:
            # wandb.init can fail (it will already have printed a message)
            return

        self._wandb_run = run
        self.run_id = "/".join((run.entity, run.project, run.id))

        # We want to only do this once
        if "job" not in config:
            job = collect_job_info()
            job["wandb_id"] = self.run_id
            job["wandb_group"] = self.group_id
            config["job"] = job
            # Update the config file with new information
            self.config_file.write_text(yaml.dump(config))
            print(config)
            self.upload_script_and_config()

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any]) -> None:
        job_type = getattr(self, "job_type", None)
        if job_type != state_dict["job_type"]:
            # Only allow resuming from the same job_type
            state_dict.pop("run_id", None)
        if job_type:
            # Typically state_dict will contain job_type="train", while we have "evaluate"
            state_dict.pop("job_type", None)
        for k, v in state_dict.items():
            setattr(self, k, v)
        self._wandb_run = None
        self._rank = torchtnt.utils.get_global_rank()

    def upload_script_and_config(self, top_secret: bool = False) -> None:
        """Uploads the script and config to W&B.

        When using `top_secret=True` only the file checksum will be uploaded.
        """
        artifact = self._wandb.Artifact(
            name=self.group_id,
            type="model",
            metadata=_simple_conf(self.read_config()),
        )
        script = self.config_file.with_suffix(".py")
        if top_secret:
            artifact.add_reference(f"file://{self.config_file}")
            artifact.add_reference(f"file://{script}")
        else:
            artifact.add_file(str(self.config_file))
            artifact.add_file(str(script))
            base_path = str(self.config_file.parent)
            self._wandb.save(str(self.config_file), policy="now", base_path=base_path)
            self._wandb.save(str(script), policy="now", base_path=base_path)
        self._wandb_run.log_artifact(artifact)

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """
        if self._rank != 0:
            return
        self.prepare()
        self._wandb_run.log(payload, step)
        print("Step:", step, {k: v for k, v in payload.items() if isinstance(v, float)})

    def close(self) -> None:
        """Close log resource, flushing if necessary.

        Logs should not be written after `close` is called.
        """
        if self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None


class LogMetrics(Callback):
    def __init__(
        self,
        logger: MetricLogger,
        frequency_steps: int = 10,
        sync_frequency: int = 100,
        max_interval: timedelta = timedelta(minutes=5),
    ):
        self.logger = logger
        self.frequency_steps = frequency_steps
        self.sync_frequency = max(sync_frequency, frequency_steps)
        self.global_rank = torchtnt.utils.get_global_rank()

        self.max_interval = max_interval
        self._last_log = datetime.now()

    def on_train_step_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed

        if step % self.frequency_steps != 0:
            if datetime.now() - self._last_log < self.max_interval:
                return
        self.log_metrics(state, step, "train/", sync=step % self.sync_frequency == 0)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed

        self.log_metrics(state, step, "train/", sync=True)

        self.logger.close()

    def on_eval_end(self, state: State, unit: TEvalUnit[Any]) -> None:
        step = (
            0
            if state.train_state is None
            else state.train_state.progress.num_steps_completed
        )

        self.log_metrics(state, step, "eval/", sync=True)

    def log_metrics(self, state: State, step: int, prefix: str, sync: bool) -> None:
        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = (
            state.eval_state.metrics if prefix == "eval/" else state.train_state.metrics  # type: ignore
        )
        wandb = isinstance(self.logger, WandbLogger)
        actual_metrics = metrics.compute(sync=sync, prefix=prefix, wandb=wandb)

        report, total_calls, total_time = torchtnt.utils.timer._make_report(state.timer)
        for row in report[:10]:
            name, avg_duration, num_calls, total_duration, percentage = row
            if percentage < 1:
                continue
            actual_metrics[f"timer/{name}"] = percentage

        if self.global_rank == 0:
            self.logger.log_dict(actual_metrics, step)

        metrics.reset()
        self._last_log = datetime.now()


class WandbCsvWriter(Callback):
    """A callback to write prediction outputs to a W&B table. This reuse the
    torchtnt.BaseCSVWriter API.

    This callback provides an interface to simplify writing outputs during prediction
    into a CSV file. This callback must be extended with an implementation for
    ``get_batch_output_rows`` to write the desired outputs as rows in the CSV file.

    By default, outputs at each step across all processes will be written into the same CSV file.
    The outputs in each row is a a list of strings, and should match
    the columns names defined in ``header_row``.

    Args:
        header_row: name of the columns
        table_name: name of the table
    """

    def __init__(
        self,
        header_row: tp.List[str],
        logger: WandbLogger,
        table_name: str,
        limit: int = 1000,
    ) -> None:
        super().__init__()
        self.columns = header_row
        self.table_name = table_name
        self.logger = logger
        self.limit = limit
        self._table_size = 0
        self._world_size: int = torchtnt.utils.distributed.get_world_size()

    def get_batch_output_rows(
        self, step_output: tp.Any
    ) -> tp.Sequence[tp.Tuple[Any, ...]]:
        if isinstance(step_output, list):
            return step_output
        raise NotImplementedError()

    def prepare(self) -> None:
        import wandb

        self.logger.prepare()
        self._table = wandb.Table(columns=self.columns, data=[])
        self._table_size = 0

    def on_eval_start(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        self.prepare()

    def on_train_start(self, state: State, unit: TTrainUnit[tp.Any]) -> None:
        self.prepare()

    def on_train_step_end(self, state: State, unit: TTrainUnit[tp.Any]) -> None:
        if self._table_size > self.limit:
            return
        assert state.train_state is not None
        step_output = state.train_state.step_output
        batch_output_rows = self.get_batch_output_rows(step_output)

        # Check whether the first item is a list or not
        for row in batch_output_rows:
            self._table.add_data(*row)
            self._table_size += 1

        if self._table_size > self.limit:
            self.logger._wandb_run.log({self.table_name: self._table}, step=0)

    def on_eval_step_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        if self._table_size > self.limit:
            return

        assert state.eval_state is not None
        step_output = state.eval_state.step_output
        batch_output_rows = self.get_batch_output_rows(step_output)

        # Check whether the first item is a list or not
        for row in batch_output_rows:
            self._table.add_data(*row)
            self._table_size += 1

    def on_eval_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        if state.train_state is None:
            step = 0
        else:
            step = state.train_state.progress.num_steps_completed
        self.logger._wandb_run.log({self.table_name: self._table}, step=step)


def _simple_conf(config: tp.Any) -> tp.Any:
    """W&B doesn't handle dataclasses or NamedTuple for config, convert them to dict."""
    if hasattr(config, "_asdict"):
        config = config._asdict()
    if dataclasses.is_dataclass(config):
        config = dataclasses.asdict(config)
    if isinstance(config, dict):
        return {k: _simple_conf(v) for k, v in config.items()}
    if isinstance(config, Path):
        config = str(config)
    if isinstance(config, torch.device):
        config = str(config)
    return config
