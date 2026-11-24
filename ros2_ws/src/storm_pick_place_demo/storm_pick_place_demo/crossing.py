"""Choose whether the arm passes above or below a human limb, from a collision forecast.

The forecast is the MarkerArray published on /rtcosmik/collision_forecast: the
human CYLINDER markers repeated for several future times, each marker's
header.stamp being its predicted time. Cylinders become capsules on the same
centre line as the planner's sphere chains (half length max(h/2 - r, 0)).

Each candidate via point is checked with a robot proxy: a vertical segment from
the gripper up to the wrist (with the gripper pointing down, the wrist and elbow
sit above the gripper). The side whose proxy keeps the largest predicted
clearance during the crossing window wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CapsuleForecast:
    times: np.ndarray  # [K] seconds, increasing
    a: np.ndarray  # [K, N, 3] centre-line endpoint
    b: np.ndarray  # [K, N, 3] centre-line endpoint
    radius: np.ndarray  # [K, N]


def forecast_from_markers(markers) -> Optional[CapsuleForecast]:
    """Group CYLINDER markers by stamp; every time step must hold the same number of capsules."""
    by_time: dict[int, list] = {}
    for m in markers:
        if m.type != m.CYLINDER or m.action != m.ADD:
            continue
        stamp_ns = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
        o = m.pose.orientation
        axis = np.array([2 * (o.x * o.z + o.y * o.w), 2 * (o.y * o.z - o.x * o.w), 1 - 2 * (o.x * o.x + o.y * o.y)])
        r = 0.5 * max(m.scale.x, m.scale.y)
        half = max(0.5 * m.scale.z - r, 0.0)
        c = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        by_time.setdefault(stamp_ns, []).append((c - half * axis, c + half * axis, r))
    if not by_time:
        return None
    stamps = sorted(by_time)
    counts = {len(by_time[s]) for s in stamps}
    if len(counts) != 1:
        raise ValueError(f"forecast time steps hold different capsule counts: {sorted(counts)}")
    return CapsuleForecast(
        times=np.asarray(stamps, dtype=np.float64) * 1e-9,
        a=np.array([[c[0] for c in by_time[s]] for s in stamps]),
        b=np.array([[c[1] for c in by_time[s]] for s in stamps]),
        radius=np.array([[c[2] for c in by_time[s]] for s in stamps]),
    )


def segment_capsule_clearance(p0, p1, robot_radius, a, b, radius, samples: int = 9) -> float:
    """Minimum surface distance between a robot segment (sampled) and capsules a-b of given radii."""
    pts = np.linspace(np.asarray(p0, float), np.asarray(p1, float), samples)  # [S, 3]
    ab = b - a  # [N, 3]
    denom = np.maximum(np.einsum("nd,nd->n", ab, ab), 1e-12)
    s = np.clip(np.einsum("snd,nd->sn", pts[:, None, :] - a[None], ab) / denom, 0.0, 1.0)  # [S, N]
    closest = a[None] + s[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - closest, axis=-1) - radius[None] - robot_radius
    return float(d.min())


def choose_side(
    forecast: CapsuleForecast,
    vias: dict[str, np.ndarray],
    t_from: float,
    t_to: float,
    wrist_height: float = 0.25,
    robot_radius: float = 0.06,
) -> tuple[str, dict[str, float]]:
    """Side whose via point keeps the largest predicted clearance over [t_from, t_to].

    Forecast times outside the window are ignored; if none fall inside, the
    nearest forecast time is used.
    """
    inside = np.where((forecast.times >= t_from) & (forecast.times <= t_to))[0]
    if inside.size == 0:
        inside = np.array([int(np.argmin(np.abs(forecast.times - 0.5 * (t_from + t_to))))])
    clearance = {}
    for side, via in vias.items():
        via = np.asarray(via, dtype=np.float64)
        top = via + np.array([0.0, 0.0, wrist_height])
        clearance[side] = min(
            segment_capsule_clearance(via, top, robot_radius, forecast.a[k], forecast.b[k], forecast.radius[k])
            for k in inside
        )
    return max(clearance, key=clearance.get), clearance
