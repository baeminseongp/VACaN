"""Small dependency-free planar homography implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math


Point = tuple[float, float]


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    size = len(target)
    augmented = [row.copy() + [target[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            raise ValueError("homography correspondences are singular")
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


@dataclass(frozen=True)
class Homography:
    values: tuple[float, float, float, float, float, float, float, float, float]

    def project(self, point: Point) -> Point:
        x, y = point
        h = self.values
        scale = h[6] * x + h[7] * y + h[8]
        if abs(scale) < 1e-12:
            raise ValueError("point projects to infinity")
        return (
            (h[0] * x + h[1] * y + h[2]) / scale,
            (h[3] * x + h[4] * y + h[5]) / scale,
        )


def fit_homography(source: list[Point], target: list[Point]) -> Homography:
    if len(source) != 4 or len(target) != 4:
        raise ValueError("exactly four point correspondences are required")
    matrix: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(source, target):
        matrix.extend([
            [u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v],
            [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v],
        ])
        values.extend([x, y])
    solution = _solve(matrix, values)
    return Homography(tuple(solution + [1.0]))  # type: ignore[arg-type]


def rmse(first: list[Point], second: list[Point]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("equal non-empty point lists are required")
    return math.sqrt(sum(math.dist(a, b) ** 2 for a, b in zip(first, second)) / len(first))
