"""Water-Filling Primal-Dual (WFPD) online MoE router.

Core algorithm for: "Online Capacitated Routing for Mixture-of-Experts:
Max-Load Bounds and Competitive Gate-Score Retention without Auxiliary Losses."

The per-expert dual price beta[e] (the "water level") rises as the expert fills,
so load balancing is an algorithmic property of routing rather than a training
penalty -- no auxiliary loss, no lambda tuning, no retraining.
"""
from __future__ import annotations
import numpy as np

__all__ = ["WaterFillingRouter"]


class WaterFillingRouter:
    """O(m + k log C) per-token primal-dual router with capacity-aware prices.

    Interface mirrors the baselines in baselines.py:
        Router(num_experts, capacity, top_k)
        route(g_t)      -> chosen expert ids        (online, single token)
        route_batch(G)  -> (n, m) binary assignment (offline driver)
        reset()
    """

    def __init__(self, num_experts: int, capacity: int, top_k: int):
        self.m = int(num_experts)
        self.C = int(capacity)
        self.k = int(top_k)
        self.gamma = np.e - 1.0                            # additive normalizer
        self.beta = np.zeros(self.m, dtype=np.float64)     # dual prices (water levels)
        self.load = np.zeros(self.m, dtype=np.int64)       # current loads

    def reset(self) -> None:
        self.beta[:] = 0.0
        self.load[:] = 0

    def route(self, g_t: np.ndarray) -> np.ndarray:
        """Route one token. g_t: (m,) nonneg gate scores -> chosen expert ids."""
        feasible = self.load < self.C
        surplus = np.where(feasible, g_t - self.beta, -np.inf)
        kth = min(self.k, self.m) - 1
        cand = np.argpartition(-surplus, kth)[: self.k]
        chosen = cand[surplus[cand] > 0]                   # only positive surplus
        for e in chosen:
            self.load[e] += 1
            self.beta[e] = self.beta[e] * (1.0 + 1.0 / self.C) \
                + g_t[e] / (self.gamma * self.C)           # exponential dual update
        return chosen.astype(np.int64)

    def route_batch(self, G: np.ndarray) -> np.ndarray:
        """Offline driver: stream rows of G, return (n, m) binary assignment."""
        self.reset()
        n, m = G.shape
        A = np.zeros((n, m), dtype=np.int64)
        for t in range(n):
            for e in self.route(G[t]):
                A[t, e] = 1
        return A


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m, k = 512, 8, 2
    C = int(np.ceil(1.25 * n * k / m))
    G = rng.gamma(shape=2.0, scale=1.0, size=(n, m))       # skewed -> imbalance pressure

    A = WaterFillingRouter(m, C, k).route_batch(G)
    load = A.sum(0)
    print(f"WFPD  obj={float((G * A).sum()):9.1f}  Lmax={int(load.max()):4d}  "
          f"served={(A.sum(1) > 0).mean():6.1%}  cap={C}")
