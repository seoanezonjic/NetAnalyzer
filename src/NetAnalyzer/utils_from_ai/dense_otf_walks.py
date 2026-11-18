#!/usr/bin/env python
from __future__ import annotations

"""
This library was made using claude code + opus
===============
The prompt:
================

This folder contains a python package called Pecanpy. Analyze the package and build a script that isolates the funcionality of the methods "from_mat" from class DenseOTF and "simulate_walks" from class Base. These methods are in the file src/pecanpy/pecanpy.py. The generated script should be compatible with numpy 2.5.4 and documented.

==============
The output:
==============
I wrote the standalone script at scripts/dense_otf_walks.py. It does what DenseOTF.from_mat and Base.simulate_walks do, and it ran with no warnings under numpy 2.5.4.

What’s in it:
--------------
- DenseOTFWalker.from_mat(adj_mat, node_ids, p, q, extend, gamma, random_state, verbose): builds the graph from a dense matrix, just like DenseOTF.from_mat. It checks the input: the matrix must be square, weights can’t be negative, NaN or infinite, and there must be one ID per row. If you leave out node_ids, the nodes are named "0" to "n-1".
simulate_walks(num_walks, walk_length): works the same way as the original. Every node starts num_walks walks, the start order is shuffled, the first step picks a neighbour in proportion to edge weight, and every later step uses the node2vec bias (or the node2vec+ rule if extend=True). A walk stops at a node with no outgoing edges, and a full walk has walk_length + 1 nodes, like in PecanPy.
- Documentation: a module docstring explains the algorithm, the output format, the differences from PecanPy and how to use it. Every class and method has its own docstring.
- Command line: --demo runs on a small built-in graph, and --input graph.dense.npz reads a PecanPy dense graph file. Each walk is written as one line.

Why NumPy 2.5.4 works
-----------------------
The original package can’t run on numpy 2.5.4 because of its other libraries: numba, nptyping and numba-progress. So the script depends on NumPy alone and uses numpy.random.Generator for randomness.

What changes compared with PecanPy
--------------------------------------
- Random numbers: the same seed always gives the same walks, but not the walks PecanPy gives for that seed. The probability of each step is identical.
- Speed: walks are generated one after another in plain Python and NumPy, without numba or several workers. 3,000 walks of length 80 on a 300-node graph took about 4 seconds.
- Rounding: the neighbour pick can’t go past the end of the neighbour list when floating-point rounding leaves the running total of probabilities just under the random draw. PecanPy has no guard for that rare case.

How I checked it
--------------------
- Probabilities: I installed the original PecanPy in a separate environment and compared step probabilities for every (current, previous) node pair on random weighted graphs. Both node2vec and node2vec+ matched to within 2e-16.
- Walks: walk counts, walk lengths and start-node counts were identical to the original. Every step followed an edge, and the sampled step frequencies agreed within sampling noise.
- Under numpy 2.5.4: the demo, a run on a .npz file and the reproducibility check all passed.

================
As opus deleted numba support, was requested to recover back 
The prompt
================
- the new script can parallelize the execution using several workers?
- why do you dropped numba ?
- Yes, use numba (when opus asked "Add numba back for compiled, parallel walks?")
===============

=============
output
=============
scripts/dense_otf_walks.py now uses numba, so walks are compiled and run in parallel on several threads. It still works without numba: it falls back to plain NumPy and generates the walks one after another. I tested it with numpy 2.5.4 and numba 0.68.1.

How it works
-------------
- One set of code for both cases. The probability calculation, the step choice and the walk loop are written so numba can compile them. When numba isn’t installed, the same functions run as ordinary Python, so the two cases can’t drift apart.
- workers=N (and --workers on the command line) sets the number of threads. None uses all cores. Asking for more than one worker without numba prints a warning.
- Same walks whatever the setup. For a given random_state, every walk gets its own seed up front. So the walks are the same with 1, 4 or 12 workers, and the same with or without numba. PecanPy seeds each thread instead (seed + thread_id), so its walks change with the number of threads.
- Compile cache: numba stores the compiled code in scripts/__pycache__, which git already ignores, so later runs start straight away.
"""
"""Standalone node2vec / node2vec+ random walks on a dense adjacency matrix.

This script isolates two pieces of PecanPy (``src/pecanpy/pecanpy.py``):

* ``DenseOTF.from_mat`` -- build a graph from a dense adjacency matrix and a
  list of node IDs (implemented in PecanPy by ``DenseGraph.from_mat``).
* ``Base.simulate_walks`` -- generate ``num_walks`` biased random walks from
  every node, with the transition probabilities of ``DenseOTF`` computed
  "on the fly" (``DenseRWGraph.get_normalized_probs`` and
  ``DenseRWGraph.get_extended_normalized_probs``).

Dependencies
------------
* NumPy (required). Works with NumPy 2.x; tested with 2.5.4.
* numba (optional, recommended). When installed, the walk loop is compiled
  and walks run in parallel on ``workers`` threads, as in PecanPy. Tested with
  numba 0.68.1, which supports NumPy up to 2.5.x. Without numba the same code
  runs as plain Python/NumPy, one walk at a time.

PecanPy's other dependencies (numba-progress, nptyping, gensim) are not used.

Algorithm
---------
Let ``W`` be the (possibly weighted, possibly directed) adjacency matrix, with
``W[i, j] != 0`` meaning there is an edge ``i -> j`` of weight ``W[i, j]``.

A walk starts at a node ``s``. The first step is a plain first-order step:
the next node is chosen among the neighbours of ``s`` with probability
proportional to the edge weight. Every later step from the current node
``c`` (having arrived from the previous node ``t``) uses the second-order
node2vec bias on the weight of each candidate edge ``c -> x``:

* ``x == t`` (go back): weight divided by the return parameter ``p``.
* ``x`` is not a neighbour of ``t`` (move outward): weight divided by the
  in-out parameter ``q``.
* ``x`` is a neighbour of ``t`` (stay close): weight unchanged.

The biased weights are normalised into probabilities and one neighbour is
sampled. With ``extend=True`` the node2vec+ rule is used instead (see
:func:`_transition_probs`), which also takes edge weights into account to
decide what counts as "outward".

A walk stops early when it reaches a node with no outgoing edges.

Output format (same as PecanPy)
-------------------------------
``simulate_walks`` returns a list of walks, each a list of node IDs
(``str``). A full walk holds ``walk_length + 1`` nodes: the start node plus
``walk_length`` steps. A walk that hits a dead end is shorter, and a walk
starting at a node without neighbours is just ``[start_node]``. The walks
come in a random order: the start nodes (each node repeated ``num_walks``
times) are shuffled.

Randomness and reproducibility
------------------------------
A ``numpy.random.Generator`` seeded with ``random_state`` shuffles the start
nodes and gives every walk its own 32-bit seed. Each walk then draws its
random numbers from a Mersenne Twister seeded with that value (numba's
``np.random`` inside the compiled code, ``numpy.random.RandomState``
without numba; both produce the same numbers). As a result, for a given
``random_state`` the walks are the same whatever the number of workers and
whether numba is installed or not.

PecanPy instead seeds each thread with ``random_state + thread_id``, so its
walks change with the number of threads. The walks here are therefore not
the same walks PecanPy produces for a seed, but each step has the same
probabilities.

Other differences from PecanPy
------------------------------
* In rare cases rounding can leave the last value of the cumulative
  distribution slightly below the random draw, which in PecanPy would pick an
  index past the end of the neighbour list. The index is clamped here.
* ``node_ids`` may be omitted, in which case nodes are named ``"0"`` to
  ``"n-1"``.
* ``workers`` really sets the number of threads (in PecanPy it is ignored
  for walk generation, which uses all of numba's threads).
* ``walk_length=0`` returns one-node walks (PecanPy writes the first step
  over its walk-length counter in that case).

Usage
-----
As a library::

    import numpy as np
    from dense_otf_walks import DenseOTFWalker

    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    g = DenseOTFWalker.from_mat(adj, ["a", "b", "c"], p=1, q=0.5,
                                workers=4, random_state=0)
    walks = g.simulate_walks(num_walks=10, walk_length=80)

From the command line::

    # run on a small built-in example graph
    python dense_otf_walks.py --demo

    # run on a PecanPy dense graph file (.npz with keys "data" and "IDs")
    python dense_otf_walks.py --input graph.dense.npz --output walks.txt \\
        --num-walks 10 --walk-length 80 --p 1 --q 1 --workers 4 \\
        --random-state 0

Each line of the output file is one walk, with node IDs separated by spaces.

With numba, the first call compiles the kernels, which takes a few seconds.
The compiled code is cached in ``__pycache__`` next to this file, so later
runs start faster.
"""

import argparse
import sys
import warnings
from typing import Callable, List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import numba
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pure NumPy fallback: the decorators do nothing
    numba = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        """Stand-in for ``numba.njit`` that returns the function unchanged."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


__all__ = ["DenseOTFWalker", "NUMBA_AVAILABLE"]


# ---------------------------------------------------------------------- #
# Kernels
#
# These functions are written in the subset of Python/NumPy that numba can
# compile. With numba they are compiled to machine code; without it they run
# as ordinary Python, giving the same results.
# ---------------------------------------------------------------------- #
@njit(nogil=True, cache=True)
def _transition_probs(data, nonzero, thresholds, p, q, extend, cur, prev):
    """Probabilities of moving from ``cur`` to each of its neighbours.

    Matches ``DenseRWGraph.get_normalized_probs`` (node2vec) and
    ``DenseRWGraph.get_extended_normalized_probs`` (node2vec+).

    node2vec: a neighbour ``x`` of ``cur`` that is not a neighbour of
    ``prev`` has its weight divided by ``q``; going back to ``prev`` has its
    weight divided by ``p``.

    node2vec+: ``x`` counts as "outward" when the edge ``prev - x`` is
    missing or weaker than prev's noise threshold. Its weight is multiplied
    by ``alpha = 1/q + (1 - 1/q) * w(prev, x) / threshold(prev)``, which
    moves smoothly from ``1/q`` (no edge to prev) to 1 (edge at the
    threshold). If the edge ``cur -> x`` is itself below cur's threshold,
    ``alpha = min(1, 1/q)``. Going back is divided by ``p`` as in node2vec.
    Like PecanPy, this assumes ``w(prev, x) == w(x, prev)`` (undirected).

    Args:
        data: Adjacency matrix, float64, shape ``(n, n)``.
        nonzero: Edge mask (``data != 0``), bool, shape ``(n, n)``.
        thresholds: node2vec+ noise thresholds, float64, shape ``(n,)``
            (ignored when ``extend`` is False).
        p: Return parameter.
        q: In-out parameter.
        extend: Use node2vec+ instead of node2vec.
        cur: Current node index.
        prev: Previous node index, or -1 on the first step (which gives
            plain first-order, weight-proportional probabilities).

    Returns:
        Probabilities in increasing order of neighbour index.

    """
    probs = data[cur].copy()

    if prev >= 0:
        if extend:
            prev_weights = data[prev]
            out_ind = nonzero[cur] & (prev_weights < thresholds)
            out_ind[prev] = False  # going back is handled by p, not q

            t = prev_weights[out_ind] / thresholds[out_ind]
            alpha = 1 / q + (1 - 1 / q) * t
            # Suppress noisy edges out of the current node.
            alpha[probs[out_ind] < thresholds[cur]] = min(1.0, 1 / q)
            probs[out_ind] *= alpha
        else:
            # Neighbours of cur that are not neighbours of prev: "outward".
            out_ind = nonzero[cur] & ~nonzero[prev]
            out_ind[prev] = False  # going back is handled by p, not q
            probs[out_ind] /= q
        probs[prev] /= p

    probs = probs[nonzero[cur]]
    return probs / probs.sum()


@njit(nogil=True, cache=True)
def _move_forward(data, nonzero, thresholds, p, q, extend, cur, prev, u):
    """Pick the next node from ``cur`` using the uniform random number ``u``.

    Picks the first neighbour whose cumulative probability is ``>= u``, as
    PecanPy's ``DenseOTF`` ``move_forward`` does.
    """
    probs = _transition_probs(data, nonzero, thresholds, p, q, extend, cur, prev)
    cdf = np.cumsum(probs)
    choice = np.searchsorted(cdf, u)
    nbrs = np.where(nonzero[cur])[0]
    if choice >= nbrs.size:  # cdf[-1] a hair below 1.0 from rounding
        choice = nbrs.size - 1
    return nbrs[choice]


@njit(nogil=True, cache=True)
def _fill_walk(row, start, u, data, nonzero, degree, thresholds, p, q, extend):
    """Generate one walk into ``row`` (the body of ``Base._random_walks``).

    Args:
        row: Output, uint32, length ``walk_length + 2``. Receives the node
            indices of the walk; its last entry receives the number of nodes
            in the walk.
        start: Start node index.
        u: ``walk_length`` uniform random numbers in ``[0, 1)``, one per
            step.
        degree: Number of outgoing edges of each node.
        Other arguments: see :func:`_transition_probs`.

    """
    walk_length = row.size - 2
    row[0] = start
    row[-1] = 1
    if walk_length == 0 or degree[start] == 0:
        return

    # First step: plain first-order random walk.
    row[1] = _move_forward(
        data, nonzero, thresholds, p, q, extend, np.int64(start), np.int64(-1), u[0]
    )
    n_nodes = 2

    # Remaining steps: second-order biased random walk.
    for j in range(2, walk_length + 1):
        cur = np.int64(row[j - 1])
        if degree[cur] == 0:
            break  # dead end
        prev = np.int64(row[j - 2])
        row[j] = _move_forward(data, nonzero, thresholds, p, q, extend, cur, prev, u[j - 1])
        n_nodes = j + 1

    row[-1] = n_nodes


@njit(parallel=True, nogil=True, cache=True)
def _walks_parallel(starts, seeds, walk_length, data, nonzero, degree, thresholds, p, q, extend):
    """Generate one walk per start node, in parallel (numba only).

    Walk ``i`` seeds numba's (per-thread) Mersenne Twister with
    ``seeds[i]``, so its result does not depend on which thread runs it.

    Returns:
        uint32 array of shape ``(len(starts), walk_length + 2)``; see
        :func:`_fill_walk` for the layout of each row.

    """
    n = starts.size
    out = np.zeros((n, walk_length + 2), dtype=np.uint32)
    n_draws = max(walk_length, 1)
    for i in prange(n):
        np.random.seed(seeds[i])
        u = np.random.random(n_draws)
        _fill_walk(out[i], starts[i], u, data, nonzero, degree, thresholds, p, q, extend)
    return out


def _walks_serial(starts, seeds, walk_length, data, nonzero, degree, thresholds, p, q, extend):
    """Pure NumPy version of :func:`_walks_parallel`, one walk at a time.

    ``numpy.random.RandomState(seed)`` produces the same numbers as numba's
    ``np.random`` after ``np.random.seed(seed)``, so both give the same walks.
    """
    n = starts.size
    out = np.zeros((n, walk_length + 2), dtype=np.uint32)
    n_draws = max(walk_length, 1)
    for i in range(n):
        u = np.random.RandomState(int(seeds[i])).random_sample(n_draws)
        _fill_walk(out[i], int(starts[i]), u, data, nonzero, degree, thresholds, p, q, extend)
    return out


# ---------------------------------------------------------------------- #
# Public class
# ---------------------------------------------------------------------- #
class DenseOTFWalker:
    """Dense graph with node2vec(+) random walks computed on the fly.

    This class does the same job as ``pecanpy.pecanpy.DenseOTF``, limited to
    building the graph from a matrix (:meth:`from_mat`) and generating walks
    (:meth:`simulate_walks`).

    Args:
        p: Return parameter. Values below 1 make the walk more likely to go
            back to the previous node; values above 1 make it less likely
            (default: 1).
        q: In-out parameter. Values below 1 push the walk outward (DFS-like);
            values above 1 keep it in the local neighbourhood (BFS-like)
            (default: 1).
        workers: Number of threads for walk generation. ``None`` uses all of
            numba's threads (by default, one per CPU core). Without numba,
            walks always run on one thread (default: ``None``).
        extend: Use the node2vec+ transition rule (default: ``False``).
        gamma: node2vec+ only. Multiplier of the standard deviation of a
            node's edge weights, added to their mean to get the node's "noisy
            edge" threshold (default: 0).
        random_state: Seed for the random number generator. ``None`` gives
            non-reproducible walks (default: ``None``).
        verbose: Print progress to stderr while generating walks
            (default: ``False``).

    Attributes:
        nodes: Node IDs; ``nodes[i]`` is the ID of row/column ``i``.
        data: Adjacency matrix as ``float64``, shape ``(n, n)``.
        nonzero: Boolean mask of the edges (``data != 0``).
        backend: ``"numba"`` or ``"numpy"``, whichever generates the walks.

    """

    def __init__(
        self,
        p: float = 1,
        q: float = 1,
        workers: Optional[int] = None,
        extend: bool = False,
        gamma: float = 0,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        if p <= 0 or q <= 0:
            raise ValueError(f"p and q must be positive, got p={p}, q={q}")
        if workers is not None and workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        if workers is not None and workers > 1 and not NUMBA_AVAILABLE:
            warnings.warn(
                "numba is not installed, so walks are generated on one thread "
                "(install numba to use several workers)",
                stacklevel=2,
            )
        self.p = float(p)
        self.q = float(q)
        self.workers = workers
        self.extend = bool(extend)
        self.gamma = float(gamma)
        self.random_state = random_state
        self.verbose = verbose
        self.backend = "numba" if NUMBA_AVAILABLE else "numpy"

        self.nodes: List[str] = []
        self.data: NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)
        self.nonzero: NDArray[np.bool_] = np.zeros((0, 0), dtype=np.bool_)
        self._degree: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Graph construction (DenseOTF.from_mat)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_mat(
        cls,
        adj_mat: ArrayLike,
        node_ids: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> "DenseOTFWalker":
        """Build a graph from a dense adjacency matrix and node IDs.

        Same as ``pecanpy.pecanpy.DenseOTF.from_mat``: the matrix is stored as
        float, its non-zero entries are the edges, and ``node_ids[i]`` names
        row/column ``i``. The matrix is used as given, so an asymmetric matrix
        is a directed graph.

        Args:
            adj_mat: Square 2D array, ``adj_mat[i, j]`` is the weight of edge
                ``i -> j`` (0 means no edge). Weights should be non-negative.
            node_ids: One ID per row. ``None`` names the nodes ``"0"`` to
                ``"n-1"``.
            **kwargs: Passed to the constructor (``p``, ``q``, ``workers``,
                ``extend``, ``gamma``, ``random_state``, ``verbose``).

        Returns:
            A new :class:`DenseOTFWalker` ready for :meth:`simulate_walks`.

        Raises:
            ValueError: If the matrix is not square, has negative or
                non-finite weights, or ``node_ids`` has the wrong length.

        """
        g = cls(**kwargs)
        g._set_matrix(adj_mat)
        g._set_node_ids(node_ids)
        return g

    def _set_matrix(self, adj_mat: ArrayLike) -> None:
        """Store the adjacency matrix, its edge mask and node degrees."""
        data = np.asarray(adj_mat, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"adj_mat must be a square 2D array, got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("adj_mat contains NaN or infinite values")
        if np.any(data < 0):
            raise ValueError("adj_mat contains negative edge weights")

        # Copy (C-contiguous) so later changes to the caller's array don't
        # affect the graph. PecanPy also copies, through ``astype(float)``.
        self.data = np.array(data, dtype=np.float64, order="C", copy=True)
        self.nonzero = self.data != 0
        self._degree = self.nonzero.sum(axis=1).astype(np.int64)

    def _set_node_ids(self, node_ids: Optional[Sequence[str]]) -> None:
        """Store node IDs, or name nodes by index when none are given."""
        n = self.data.shape[0]
        if node_ids is None:
            self.nodes = [str(i) for i in range(n)]
            return
        node_ids = list(node_ids)
        if len(node_ids) != n:
            raise ValueError(f"Got {len(node_ids)} node IDs for a {n}x{n} matrix")
        self.nodes = node_ids

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of non-zero entries of the adjacency matrix."""
        return int(self.nonzero.sum())

    # ------------------------------------------------------------------ #
    # Transition probabilities (DenseRWGraph)
    # ------------------------------------------------------------------ #
    def noise_thresholds(self) -> NDArray[np.float64]:
        """Per-node threshold for "noisy" (weak) edges, used by node2vec+.

        For node ``i`` this is ``mean + gamma * std`` of the weights of its
        edges, floored at 0. Nodes without edges get 0. Matches
        ``DenseRWGraph.get_noise_thresholds``.
        """
        thresholds = np.zeros(self.num_nodes, dtype=np.float64)
        for i in range(self.num_nodes):
            if self._degree[i]:
                weights = self.data[i, self.nonzero[i]]
                thresholds[i] = weights.mean() + self.gamma * weights.std()
        # PecanPy stores these as float32; keep the same precision so results
        # agree with it.
        return np.maximum(thresholds, 0).astype(np.float32).astype(np.float64)

    def transition_probs(self, cur_idx: int, prev_idx: Optional[int] = None) -> NDArray[np.float64]:
        """Probabilities of moving from ``cur_idx`` to each of its neighbours.

        Args:
            cur_idx: Current node index (must have at least one neighbour).
            prev_idx: Previous node index, or ``None`` for the first step.

        Returns:
            Probabilities in increasing order of neighbour index.

        """
        thresholds = self._thresholds_for_walks()
        prev = -1 if prev_idx is None else int(prev_idx)
        return _transition_probs(
            self.data, self.nonzero, thresholds, self.p, self.q, self.extend, int(cur_idx), prev
        )

    def _thresholds_for_walks(self) -> NDArray[np.float64]:
        """Noise thresholds for node2vec+, or zeros (unused) for node2vec."""
        if self.extend:
            return self.noise_thresholds()
        return np.zeros(self.num_nodes, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Walk generation (Base.simulate_walks)
    # ------------------------------------------------------------------ #
    def simulate_walks(self, num_walks: int, walk_length: int) -> List[List[str]]:
        """Generate ``num_walks`` walks starting from every node.

        Same behaviour as ``pecanpy.pecanpy.Base.simulate_walks`` on a
        ``DenseOTF`` graph: every node is used as a start node ``num_walks``
        times, the start nodes are shuffled, and one walk is generated from
        each. With numba the walks run in parallel on ``workers`` threads.

        Args:
            num_walks: Number of walks starting from each node.
            walk_length: Number of steps per walk. A full walk has
                ``walk_length + 1`` nodes, including the start node.

        Returns:
            A list of ``num_walks * num_nodes`` walks, each a list of node
            IDs. Walks that reach a node without outgoing edges stop there.

        """
        if num_walks < 0 or walk_length < 0:
            raise ValueError("num_walks and walk_length must be non-negative")

        rng = np.random.default_rng(self.random_state)
        starts = np.tile(np.arange(self.num_nodes, dtype=np.uint32), num_walks)
        rng.shuffle(starts)  # same as PecanPy; keeps the output unordered
        seeds = rng.integers(0, 2**32, size=starts.size, dtype=np.uint32)
        total = starts.size
        if total == 0:
            return []

        kernel_args = (
            int(walk_length),
            self.data,
            self.nonzero,
            self._degree,
            self._thresholds_for_walks(),
            self.p,
            self.q,
            self.extend,
        )
        kernel: Callable = _walks_parallel if NUMBA_AVAILABLE else _walks_serial

        # With verbose, run in about 20 batches to report progress between
        # them; otherwise run everything at once.
        batch = max(total // 20, 1) if self.verbose else total
        walk_mats = []
        with _num_threads(self.workers):
            for lo in range(0, total, batch):
                hi = min(lo + batch, total)
                walk_mats.append(kernel(starts[lo:hi], seeds[lo:hi], *kernel_args))
                if self.verbose:
                    print(f"\rGenerated {hi}/{total} walks", end="", file=sys.stderr)
        if self.verbose:
            print(file=sys.stderr)

        # Map node indices back to node IDs; the last column is the length.
        nodes = self.nodes
        return [
            [nodes[k] for k in row[: row[-1]].tolist()]
            for walk_mat in walk_mats
            for row in walk_mat
        ]


class _num_threads:
    """Context manager that sets numba's thread count, then restores it."""

    def __init__(self, workers: Optional[int]):
        self.workers = workers
        self.previous: Optional[int] = None

    def __enter__(self):
        if NUMBA_AVAILABLE and self.workers is not None:
            self.previous = numba.get_num_threads()
            numba.set_num_threads(min(self.workers, numba.config.NUMBA_NUM_THREADS))
        return self

    def __exit__(self, *exc):
        if self.previous is not None:
            numba.set_num_threads(self.previous)
        return False


# ---------------------------------------------------------------------- #
# Command-line interface
# ---------------------------------------------------------------------- #
def _demo_graph() -> tuple[NDArray[np.float64], List[str]]:
    """Two weighted triangles joined by one weak edge, plus an isolated node."""
    adj = np.zeros((7, 7))
    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0),
             (3, 4, 2.0), (3, 5, 2.0), (4, 5, 2.0),
             (2, 3, 0.1)]
    for i, j, w in edges:
        adj[i, j] = adj[j, i] = w
    return adj, ["a", "b", "c", "d", "e", "f", "isolated"]


def _load_npz(path: str, weighted: bool) -> tuple[NDArray[np.float64], Optional[List[str]]]:
    """Load a PecanPy dense ``.npz`` graph (keys ``data`` and optional ``IDs``)."""
    with np.load(path, allow_pickle=False) as raw:
        data = np.asarray(raw["data"], dtype=np.float64)
        ids = [str(i) for i in raw["IDs"]] if "IDs" in raw.files else None
    if not weighted:
        data = (data != 0).astype(np.float64)
    return data, ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, generate walks and write them out."""
    parser = argparse.ArgumentParser(
        description="Generate node2vec(+) random walks from a dense adjacency matrix.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="PecanPy dense graph .npz (keys 'data', 'IDs')")
    src.add_argument("--demo", action="store_true", help="use a small built-in graph")
    parser.add_argument("--output", help="output file (default: stdout)")
    parser.add_argument("--unweighted", action="store_true",
                        help="treat every edge as weight 1")
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=80)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=None,
                        help="threads for walk generation (default: all cores; needs numba)")
    parser.add_argument("--extend", action="store_true", help="use node2vec+")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        adj, ids = _demo_graph()
    else:
        adj, ids = _load_npz(args.input, weighted=not args.unweighted)

    g = DenseOTFWalker.from_mat(
        adj, ids, p=args.p, q=args.q, workers=args.workers, extend=args.extend,
        gamma=args.gamma, random_state=args.random_state, verbose=args.verbose,
    )
    if args.verbose:
        print(f"Backend: {g.backend}", file=sys.stderr)
    walks = g.simulate_walks(args.num_walks, args.walk_length)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for walk in walks:
            out.write(" ".join(walk) + "\n")
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
