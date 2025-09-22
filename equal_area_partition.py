from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

EARTH_RADIUS_KM: float = 6371.0


def _normalise_bounds(bounds: Sequence[float]) -> Tuple[float, float]:
    """Return the sorted lower/upper pair for a numeric interval."""

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


@dataclass(frozen=True)
class EqualAreaCell:

    cell_id: int
    i: int
    j: int
    latitude: float
    longitude: float
    lat_step: float
    lon_step: float


@dataclass(frozen=True)
class EqualAreaGrid:

    n: int  # latitude rows
    m: List[int]  # longitude column counts per row
    centres: List[Tuple[float, float]]
    cells: List[EqualAreaCell]
    lat_step: float
    lon_steps: List[float]
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

    if delta_a <= 0 or delta_b <= 0:
        raise ValueError("delta_a and delta_b must be positive values (kilometres).")

    lat_min, lat_max = _normalise_bounds(lat_range)
    lon_min, lon_max = _normalise_bounds(lon_range)

    phi_min = math.radians(lat_min) 
    phi_max = math.radians(lat_max)
    lambda_min = math.radians(lon_min)
    lambda_max = math.radians(lon_max)

    lat_span_rad = abs(phi_max - phi_min)   #纬度跨度（弧度）
    lon_span_rad = abs(lambda_max - lambda_min) #经度跨度（弧度）
    if lat_span_rad <= 0 or lon_span_rad <= 0:
        raise ValueError("Latitude and longitude spans must be non-zero.")

    lat_distance = radius_km * lat_span_rad
    row_count = max(1, int(math.ceil(lat_distance / delta_a))) #行数
    lat_step_rad = lat_span_rad / row_count  #纬度步长（弧度）
    lat_step_deg = math.degrees(lat_step_rad) #纬度步长（度）

    centres: List[Tuple[float, float]] = [] 
    cells: List[EqualAreaCell] = [] 
    col_counts: List[int] = []
    lon_steps: List[float] = []

    lon_span_deg = lon_max - lon_min

    for row_idx in range(row_count):
        lat_centre = lat_min + (row_idx + 0.5) * lat_step_deg #纬度中心
        lat_centre_rad = math.radians(lat_centre) #纬度中心（弧度）
        cos_phi = max(0.0, math.cos(lat_centre_rad)) #纬度中心的余弦值
        parallel_radius = radius_km * cos_phi #该纬度行的平行圈半径
        lon_distance = parallel_radius * lon_span_rad #该纬度行的东西向距离
        if lon_distance <= 0:
            col_count = 1
        else:
            col_count = max(1, int(math.ceil(lon_distance / delta_b))) #该纬度行的列数
        lon_step_deg = lon_span_deg / col_count if col_count > 0 else lon_span_deg #该纬度行的经度步长（度）

        col_counts.append(col_count)
        lon_steps.append(lon_step_deg)

        for col_idx in range(col_count):
            lon_centre = lon_min + (col_idx + 0.5) * lon_step_deg #经度中心
            cell_id = len(cells) + 1  #从1开始的单元格ID
            centres.append((lat_centre, lon_centre))
            cells.append(
                EqualAreaCell(
                    cell_id=cell_id,
                    i=row_idx + 1, #行索引从1开始
                    j=col_idx + 1,
                    latitude=lat_centre,  #纬度中心
                    longitude=lon_centre,
                    lat_step=lat_step_deg, #纬度步长（度）
                    lon_step=lon_step_deg, 
                )
            )

    return EqualAreaGrid(
        n=row_count, #行数
        m=col_counts, #每行的列数
        centres=centres,
        cells=cells, #对应的 EqualAreaCell 对象,保存 cell_id、i/j 索引以及该格子的中心坐标
        lat_step=lat_step_deg,
        lon_steps=lon_steps, #每行的经度步长
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
