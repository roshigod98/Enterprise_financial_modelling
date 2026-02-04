from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemandCurve:
    p1: float = 7.0
    u1: int = 300
    p2: float = 10.0
    u2: int = 200
    min_units: int = 0
    max_units: int = 500

    def predict_units(self, price: float) -> int:
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
