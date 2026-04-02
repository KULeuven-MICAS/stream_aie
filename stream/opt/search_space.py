from __future__ import annotations

from dataclasses import dataclass, field

from stream.datatypes import LayerDim


@dataclass(frozen=True)
class TileSizeOption:
    """A single candidate tile size for a unique workload dimension."""

    dim: LayerDim
    tile: int
    workload_size: int


@dataclass
class SearchSpace:
    """Holds valid tile candidates per unique workload dimension.

    Extensible for future optimization variables beyond tile sizes.
    """

    options: dict[LayerDim, list[TileSizeOption]] = field(default_factory=dict)

    def add(self, dim: LayerDim, option: TileSizeOption) -> None:
        self.options.setdefault(dim, []).append(option)

    def get(self, dim: LayerDim) -> list[TileSizeOption]:
        return self.options.get(dim, [])

    def dims(self) -> list[LayerDim]:
        return list(self.options.keys())

    def is_empty(self) -> bool:
        return not self.options or all(not v for v in self.options.values())
