# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    config: dict[str, Any]

    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None:
        for inp, out, lab, score in samples:
            print(f"[{table_name}]\n[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n")


@dataclass
class FileGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None:
        filename = "generations.log" if table_name == "val/generations" else f"{table_name.replace('/', '_')}.log"
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], filename), "a") as f:
            for inp, out, lab, score in samples:
                f.write(f"[{table_name}]\n[step] {step}\n[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None:
        columns = ["step", "sample_idx", "input", "output", "label", "score"]

        if not hasattr(self, "tables"):
            self.tables = {}

        if table_name not in self.tables:
            self.tables[table_name] = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data.
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.tables[table_name].data)

        for sample_idx, sample in enumerate(samples):
            new_table.add_data(step, sample_idx, *sample)

        wandb.log({table_name: new_table}, step=step)
        self.tables[table_name] = new_table


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({table_name: swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: Optional[dict[str, Any]] = None):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger](config))

    def log(self, samples: List[Tuple[str, str, str, Any]], step: int, table_name: str = "val/generations") -> None:
        for logger in self.loggers:
            logger.log(samples, step, table_name=table_name)
