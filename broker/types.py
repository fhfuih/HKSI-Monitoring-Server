from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple, Optional

import numpy.typing as npt

ModelActionType = Enum("ThreadActionType", ["Start", "Frame", "End"])

ModelRawResult = Optional[dict]


OnDataCallback = Callable[[Optional[dict]], None]


@dataclass
class SessionAsset:
    sid: Hashable
    ending: bool = False
    on_intermediate_data: Optional[OnDataCallback] = None
    on_end_data: Optional[OnDataCallback] = None


class ModelAction(NamedTuple):
    type: ModelActionType
    sid: Hashable
    timestamp: Optional[int]
    data: Optional[npt.NDArray]


class ModelActionWithProgress(NamedTuple):
    action: ModelAction
    progress: int


class ModelResultReport(NamedTuple):
    sid: Hashable
    model_index: int
    progress: int
    result: ModelRawResult
    is_final: bool
