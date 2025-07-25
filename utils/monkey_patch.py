from typing import Callable, Concatenate, ParamSpec, Type, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
MonkeyPatchedMethod = Callable[Concatenate[T, P], R]
MonkeyPatchDecorator = Callable[[MonkeyPatchedMethod], MonkeyPatchedMethod]


def monkeypatch_method(cls: Type[T]) -> MonkeyPatchDecorator:
    def decorator(func: MonkeyPatchedMethod) -> MonkeyPatchedMethod:
        setattr(cls, func.__name__, func)
        return func

    return decorator
