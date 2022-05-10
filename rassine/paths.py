# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import ClassVar, Union

# from configpile import Parser, parsers


# def _path_is_absolute(p: Path) -> bool:
#     return p.is_absolute()


# def _path_is_relative(p: Path) -> bool:
#     return not p.is_absolute()


# @dataclass(frozen=True)
# class RelPath:
#     path: Path

#     @staticmethod
#     def from_str(s: str) -> RelPath:
#         return RelPath(Path(s))

#     def __post_init__(self) -> None:
#         assert not self.path.is_absolute(), "Relative paths cannot be absolute"

#     parser: ClassVar[Parser[RelPath]] = parsers.path.validated(
#         _path_is_relative, "Relative path cannot be absolute"
#     ).map(lambda p: RelPath(p))


# _RelPathLike = Union[RelPath, str]


# @dataclass(frozen=True)
# class RootPath:
#     path: Path

#     def __post_init__(self) -> None:
#         assert self.path.is_absolute(), "Root path must be absolute"

#     def at(self, *rel_paths: _RelPathLike) -> Path:
#         res: Path = self.path
#         for r in rel_paths:
#             if isinstance(r, RelPath):
#                 res = res / r.path
#             elif isinstance(r, str):
#                 res = res / r
#             else:
#                 raise ValueError(f"Cannot use type {type(r)}")
#         return res

#     parser: ClassVar[Parser[RootPath]] = parsers.path.validated(
#         _path_is_absolute, "Root path must be absolute"
#     ).map(lambda p: RootPath(p))
