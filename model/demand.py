from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class DemandCurve:
    p1: float = 7.0
    u1: int = 300
    p2: float = 10.0
    u2: int = 200
    min_units: int = 0
    max_units: int = 500

    def __post_init__(self) -> None:
        if any(not isfinite(v) or v < 0 for v in (self.p1, self.p2)):
            raise ValueError("Anchor prices must be finite and nonnegative")
        if any(not isinstance(v, Integral) or v < 0 for v in (self.u1, self.u2, self.min_units, self.max_units)):
            raise ValueError("Demand quantities must be nonnegative integers")
        if self.min_units > self.max_units:
            raise ValueError("Minimum demand cannot exceed maximum demand")
        if self.p1 == self.p2 and self.u1 != self.u2:
            raise ValueError("Equal anchor prices require equal quantities")

    def predict_units(self, price: float) -> int:
        if not isfinite(price) or price < 0:
            raise ValueError("Price must be finite and nonnegative")
        if self.p2 == self.p1:
            u = self.u1
        else:
            m = (self.u2 - self.u1) / (self.p2 - self.p1)
            u = self.u1 + m * (price - self.p1)

        u_int = int(round(u))
        if u_int < self.min_units:
            u_int = self.min_units
        if u_int > self.max_units:
            u_int = self.max_units
        return u_int
