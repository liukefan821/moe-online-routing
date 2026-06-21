"""Baseline routers for offline evaluation of online MoE routing.

Each router mirrors WaterFillingRouter's constructor
    Router(num_experts, capacity, top_k)
and exposes a uniform offline entry point
    route_batch(G) -> A
where G is (n, m) nonnegative gate scores and A is an (n, m) binary assignment
matrix (A[t, e] == 1 iff token t is dispatched to expert e).

TokenChoice is genuinely online and also exposes route(g_t)/reset() in the
WFPD style; ExpertChoice and LPR are batch by construction.
"""
from __future__ import annotations
import numpy as np

__all__ = ["TokenChoiceRouter", "ExpertChoiceRouter", "LPRRouter"]


def _top_k_indices(row: np.ndarray, k: int) -> np.ndarray:
    k = min(k, row.shape[0])
    idx = np.argpartition(-row, k - 1)[:k]
    return idx[np.argsort(-row[idx])]  # desc order -> deterministic ties


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2.0 * cum.sum() / cum[-1]) / n)


class TokenChoiceRouter:
    """GShard/Switch default: each token greedily requests its top-k experts;
    a request is honored only while the expert has a free slot, else dropped.
    Stream order decides contested slots, so this is genuinely online."""

    def __init__(self, num_experts: int, capacity: int, top_k: int):
        self.m, self.C, self.k = int(num_experts), int(capacity), int(top_k)
        self.load = np.zeros(self.m, dtype=np.int64)

    def reset(self) -> None:
        self.load[:] = 0

    def route(self, g_t: np.ndarray) -> np.ndarray:
        chosen = []
        for e in _top_k_indices(g_t, self.k):
            if self.load[e] < self.C:
                self.load[e] += 1
                chosen.append(int(e))
        return np.asarray(chosen, dtype=np.int64)

    def route_batch(self, G: np.ndarray) -> np.ndarray:
        self.reset()
        n, m = G.shape
        A = np.zeros((n, m), dtype=np.int64)
        for t in range(n):
            for e in self.route(G[t]):
                A[t, e] = 1
        return A


class ExpertChoiceRouter:
    """Expert-choice routing (Zhou et al., 2022): each expert selects its top-C
    tokens by gate score. Perfect load balance by construction; a token may get
    a variable number of experts. Batch by nature."""

    def __init__(self, num_experts: int, capacity: int, top_k: int):
        self.m, self.C, self.k = int(num_experts), int(capacity), int(top_k)

    def route_batch(self, G: np.ndarray) -> np.ndarray:
        n, m = G.shape
        A = np.zeros((n, m), dtype=np.int64)
        C = min(self.C, n)
        for e in range(m):
            col = G[:, e]
            top = np.arange(n) if C >= n else np.argpartition(-col, C - 1)[:C]
            A[top, e] = 1
        return A


class LPRRouter:
    """Balanced-assignment router approximating Latent Prototype Routing
    (Wu et al., 2025). LPR's *effect* is a near-uniform expert load via a
    balanced (optimal-transport) assignment; we realize that mechanism directly
    with entropic OT (Sinkhorn) on the gate scores, token marginal k and expert
    marginal n*k/m, then dispatch each token's top experts under the transport
    plan subject to capacity. A learning-free surrogate for the routing
    DECISION (not the learned prototypes); strong load-balancing baseline."""

    def __init__(self, num_experts: int, capacity: int, top_k: int,
                 tau: float = 1.0, n_iters: int = 50, eps: float = 1e-9):
        self.m, self.C, self.k = int(num_experts), int(capacity), int(top_k)
        self.tau, self.n_iters, self.eps = float(tau), int(n_iters), float(eps)

    def _sinkhorn(self, G: np.ndarray) -> np.ndarray:
        n, m = G.shape
        K = np.exp((G - G.max()) / self.tau)              # stabilized kernel
        r = np.full(n, self.k, dtype=np.float64)          # k units per token
        c = np.full(m, n * self.k / m, dtype=np.float64)  # balanced expert mass
        u, v = np.ones(n), np.ones(m)
        for _ in range(self.n_iters):
            u = r / (K @ v + self.eps)
            v = c / (K.T @ u + self.eps)
        return (u[:, None] * K) * v[None, :]

    def route_batch(self, G: np.ndarray) -> np.ndarray:
        n, m = G.shape
        P = self._sinkhorn(G.astype(np.float64))
        A = np.zeros((n, m), dtype=np.int64)
        load = np.zeros(m, dtype=np.int64)
        for t in np.argsort(-P.max(axis=1)):              # confident tokens first
            picked = 0
            for e in np.argsort(-P[t]):
                if picked == self.k:
                    break
                if load[e] < self.C:
                    A[t, e] = 1
                    load[e] += 1
                    picked += 1
        return A


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m, k = 512, 8, 2
    C = int(np.ceil(1.25 * n * k / m))
    G = rng.gamma(shape=2.0, scale=1.0, size=(n, m))      # skewed -> imbalance pressure

    def report(name: str, A: np.ndarray) -> None:
        load = A.sum(0)
        print(f"{name:18s} obj={float((G * A).sum()):9.1f}  "
              f"Lmax={int(load.max()):4d}  Gini={_gini(load):.3f}  "
              f"served={(A.sum(1) > 0).mean():6.1%}")

    for R in (TokenChoiceRouter, ExpertChoiceRouter, LPRRouter):
        report(R.__name__, R(m, C, k).route_batch(G))
