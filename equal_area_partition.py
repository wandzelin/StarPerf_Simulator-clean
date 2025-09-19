from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

EARTH_RADIUS_KM: float = 6371.0
#校验经纬度边界
def _normalise_bounds(bounds: Sequence[float]) -> Tuple[float, float]:
    try:
        lower, upper = bounds  # type: ignore[misc]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
        raise ValueError("Bounds must contain exactly two numeric values.") from exc

    lower = float(lower)
    upper = float(upper)
    if lower == upper:
        raise ValueError("Bounds must span a non-zero interval.")
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper

#输入经纬度边界，计算小区的行列数和经纬度间隔
def _cell_count_and_step(lower: float, upper: float, delta: float) -> Tuple[int, float]:
    if delta <= 0:
        raise ValueError("Cell size must be a positive value.")
    span = upper - lower
    if span <= 0:
        raise ValueError("Upper bound must be greater than lower bound.")

    count = max(1, int(math.ceil(span / delta)))
    step = span / count
    return count, step

@dataclass(frozen=True)
class EqualAreaCell:
    #表示单个小区的编号、网格索引与中心点坐标。

    cell_id: int
    i: int
    j: int
    latitude: float
    longitude: float

@dataclass(frozen=True)
class EqualAreaGrid:
    m: int
    n: int
    centres: List[Tuple[float, float]]
    cells: List[EqualAreaCell]
    step_a: float
    step_b: float
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    radius_km: float


def approximate_equal_area_grid(
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    delta_a: float,
    delta_b: float,
    radius_km: float = EARTH_RADIUS_KM,
) -> EqualAreaGrid:
    lat_min, lat_max = _normalise_bounds(lat_range)
    lon_min, lon_max = _normalise_bounds(lon_range)

    lambda_min = math.radians(lon_min)
    lambda_max = math.radians(lon_max)
    phi_min = math.radians(lat_min)
    phi_max = math.radians(lat_max)

    a_min = radius_km * lambda_min
    a_max = radius_km * lambda_max
    b_min = radius_km * math.sin(phi_min)
    b_max = radius_km * math.sin(phi_max)

    n, step_a = _cell_count_and_step(a_min, a_max, delta_a)
    m, step_b = _cell_count_and_step(b_min, b_max, delta_b)

    centres: List[Tuple[float, float]] = []
    cells: List[EqualAreaCell] = []
    for i in range(m):
        # i 按照由南向北（下到上）的顺序遍历纬度带
        b_centre = b_min + (i + 0.5) * step_b
        sin_phi = max(-1.0, min(1.0, b_centre / radius_km))
        lat_centre = math.degrees(math.asin(sin_phi))
        for j in range(n):
            # j 按照由西向东（左到右）的顺序遍历经度带
            a_centre = a_min + (j + 0.5) * step_a
            lon_centre = math.degrees(a_centre / radius_km)
            centres.append((lat_centre, lon_centre))
            cell_id = i * n + j + 1  # 以 (i, j) 的行优先顺序生成 1 起编号
            cells.append(
                EqualAreaCell(
                    cell_id=cell_id,
                    i=i + 1,
                    j=j + 1,
                    latitude=lat_centre,
                    longitude=lon_centre,
                )
            )

    return EqualAreaGrid(
        m=m,
        n=n,
        centres=centres,
        cells=cells,
        step_a=step_a,
        step_b=step_b,
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        radius_km=radius_km,
    )


__all__ = [
    "EARTH_RADIUS_KM",
    "EqualAreaCell",
    "EqualAreaGrid",
    "approximate_equal_area_grid",
]