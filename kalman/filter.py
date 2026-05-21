
import numpy as np
import pandas as pd


class MacroKalman2D:
    """Two-dimensional Kalman filter with price and hidden velocity states."""

    def __init__(
        self,
        initial_price: float,
        rho: float = 0.86,
        alpha: float = 0.25,
        q: float = 0.08,
        r: float = 2.5,
    ) -> None:
        self.x = np.array([[float(initial_price)], [0.0]], dtype=float)
        self.p = np.eye(2, dtype=float)
        self.rho = float(np.clip(rho, 0.0, 1.0))
        self.alpha = float(alpha)
        self.q = max(float(q), 1e-9)
        self.r = max(float(r), 1e-9)

        self.f = np.array([[1.0, 1.0], [0.0, self.rho]], dtype=float)
        self.b = np.array([[0.5 * self.alpha], [self.alpha]], dtype=float)
        self.h = np.array([[1.0, 0.0]], dtype=float)
        self.q_matrix = np.array([[self.q, 0.0], [0.0, self.q]], dtype=float)
        self.i = np.eye(2, dtype=float)

    def predict(self, u_score: float = 0.0) -> None:
        u = 0.0 if pd.isna(u_score) else float(u_score)
        self.x = self.f @ self.x + self.b * u
        self.p = self.f @ self.p @ self.f.T + self.q_matrix

    def update(self, observed_price: float | None, vol_ratio: float = 1.0) -> None:
        if observed_price is None or pd.isna(observed_price):
            return

        safe_ratio = 1.0 if pd.isna(vol_ratio) else max(float(vol_ratio), 0.05)
        r_dynamic = self.r * safe_ratio
        r_matrix = np.array([[r_dynamic]], dtype=float)
        z = np.array([[float(observed_price)]], dtype=float)
        residual = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + r_matrix
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ residual
        self.p = (self.i - k @ self.h) @ self.p

    def step(
        self,
        observed_price: float | None,
        u_score: float = 0.0,
        vol_ratio: float = 1.0,
    ) -> tuple[float, float]:
        self.predict(u_score)
        self.update(observed_price, vol_ratio=vol_ratio)
        return float(self.x[0, 0]), float(self.x[1, 0])


def run_kalman(
    price: pd.Series,
    u_score: pd.Series,
    vol_ratio: pd.Series,
    rho: float,
    alpha: float,
    q: float,
    r: float,
) -> pd.DataFrame:
    observed = pd.to_numeric(price, errors="coerce").dropna()
    if observed.empty:
        return pd.DataFrame()

    u_aligned = pd.to_numeric(u_score.reindex(observed.index), errors="coerce").fillna(0.0)
    vol_aligned = pd.to_numeric(vol_ratio.reindex(observed.index), errors="coerce").fillna(1.0)
    model = MacroKalman2D(
        initial_price=float(observed.iloc[0]),
        rho=rho,
        alpha=alpha,
        q=q,
        r=r,
    )

    rows: list[dict[str, float]] = []
    for timestamp, value in observed.items():
        estimated_price, velocity = model.step(
            value,
            u_aligned.loc[timestamp],
            vol_ratio=vol_aligned.loc[timestamp],
        )
        rows.append(
            {
                "timestamp": timestamp,
                "observed": float(value),
                "kalman_price": estimated_price,
                "velocity": velocity,
                "U_score": float(u_aligned.loc[timestamp]),
                "vol_ratio": float(vol_aligned.loc[timestamp]),
            }
        )

    result = pd.DataFrame(rows).set_index("timestamp")
    previous_velocity = result["velocity"].shift(1)
    result["buy_signal"] = (result["velocity"] > 0.0) & (previous_velocity <= 0.0)
    result["sell_signal"] = (result["velocity"] < 0.0) & (previous_velocity >= 0.0)
    return result
