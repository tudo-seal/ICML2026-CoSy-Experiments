from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from cosy.core import SpecificationBuilder
from cosy.core.types import Constructor, DataGroup, Literal, Var
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository


# ---------------------------------------------------------------------------
#  New component labels (module-level so they pickle cleanly)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sine:
    trivial = ()


@dataclass(frozen=True)
class Cosine:
    trivial = ()


# ---------------------------------------------------------------------------
#  Torch modules (w -> w elementwise, like SynthTanh)
# ---------------------------------------------------------------------------
class SynthSine(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.o = output_dim

    def forward(self, x):
        return torch.sin(x)


class SynthCosine(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.o = output_dim

    def forward(self, x):
        return torch.cos(x)


NEW_COMPONENTS = {"sine": (Sine, SynthSine), "cosine": (Cosine, SynthCosine)}


# ---------------------------------------------------------------------------
#  Repository extension (Gamma + Delta)
# ---------------------------------------------------------------------------
class ExtendedDAMGrepository(DAMGrepository):
    """DAMGrepository plus sine/cosine, added externally via subclassing."""

    class Label(DAMGrepository.Label):
        def iter_sine(self):
            yield Sine()

        def iter_cosine(self):
            yield Cosine()

        def __iter__(self):
            yield from super().__iter__()
            yield from self.iter_sine()
            yield from self.iter_cosine()

        def __contains__(self, item):
            return isinstance(item, (Sine, Cosine)) or super().__contains__(item)

    def _activation_entry(self, name, label_iter, labels, para_labels, dimension):
        """SpecificationBuilder entry for a w->w component (tanh blueprint)."""
        b = (SpecificationBuilder()
             .parameter("l", labels, lambda v: list(label_iter()))
             .parameter("i", dimension)
             .parameter("o", dimension, lambda v: [v["i"]]))
        projections = [
            lambda v: [(v["l"], v["i"], v["o"])],
            lambda v: [(v["l"], v["i"], None)],
            lambda v: [(v["l"], None, v["o"])],
            lambda v: [(None, v["i"], v["o"])],
            lambda v: [(v["l"], None, None)],
            lambda v: [(None, None, v["o"])],
            lambda v: [(None, v["i"], None)],
        ]
        for idx, proj in enumerate(projections, start=1):
            b = b.parameter(f"para{idx}", para_labels, proj)
        suffix = Constructor("DAG_component",
                             Constructor("input", Var("i"))
                             & Constructor("input", Literal(None))
                             & Constructor("output", Var("o"))
                             & Constructor("output", Literal(None))
                             & Constructor("structure", Var("para1"))
                             & Constructor("structure", Var("para2"))
                             & Constructor("structure", Var("para3"))
                             & Constructor("structure", Var("para4"))
                             & Constructor("structure", Var("para5"))
                             & Constructor("structure", Var("para6"))
                             & Constructor("structure", Var("para7"))
                             & Constructor("structure", Literal(None))
                             ) & Constructor("non_ID")
        return b.suffix(suffix)

    def specification(self):
        spec = super().specification()   # uses self.Label -> extended labels in Delta
        labels = self.Label(self.dimensions, self.linear_feature_dimensions,
                            self.constant_values)
        para_labels = self.Para(labels, self.dimensions)
        dimension = DataGroup("dimension", self.dimensions)
        spec["sine"] = self._activation_entry(
            "sine", labels.iter_sine, labels, para_labels, dimension)
        spec["cosine"] = self._activation_entry(
            "cosine", labels.iter_cosine, labels, para_labels, dimension)
        return spec


# ---------------------------------------------------------------------------
#  Algebra extensions
# ---------------------------------------------------------------------------
def extend_algebra(base: dict, kind: str = "module") -> dict:
    """Add sine/cosine entries to a base algebra dict.

    kind='module'  : pytorch_function_algebra / pytorch_model_algebra
    kind='name'    : algebras representing components by name/tuple
                     (pretty_term, request, refinement, hierarchy name level)
    """
    alg = dict(base)
    for name, (_, module_cls) in NEW_COMPONENTS.items():
        if kind == "module":
            alg[name] = (lambda l, i, o, p1, p2, p3, p4, p5, p6, p7,
                         _m=module_cls: _m(o))
        elif kind == "name":
            alg[name] = (lambda l, i, o, p1, p2, p3, p4, p5, p6, p7,
                         _n=name: f"({_n}, {i}, {o})")
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return alg


# Leaf combinators whose histogram output is a one-hot over the op basis; all
# other entries (before/beside/learner/loss/optimizer/...) combine children.
_HIST_LEAVES = ("edges", "swap", "linear_layer", "sigmoid", "relu", "tanh",
                "sum", "product", "copy")


def _padded_like(f, n_new: int):
    """Wrap f to append n_new zeros, PRESERVING its exact positional arity.

    cosy's Tree.interpret inspects signatures (counts non-default parameters)
    to apply children, so *args wrappers or array-valued defaults break it.
    """
    import inspect
    n_params = len(inspect.signature(f).parameters)
    args = ", ".join(f"a{k}" for k in range(n_params))
    ns = {"f": f, "np": np, "zeros": np.zeros(n_new)}
    exec(f"def w({args}):\n    return np.concatenate([f({args}), zeros])", ns)
    return ns["w"]


def _one_hot_entry(vec: np.ndarray):
    """10-positional-arg entry returning a fixed one-hot (closure, no defaults)."""
    def entry(l, i, o, p1, p2, p3, p4, p5, p6, p7):
        return vec.copy()
    return entry


def extend_histogram_algebra(base: dict) -> dict:
    """Extend the DAMG-kernel feature basis from 9 to 9+len(NEW_COMPONENTS).

    Pads every LEAF one-hot with zeros and appends one-hots for the new ops.
    Composite entries pass through: they only combine (already padded) child
    vectors. Verify shape consistency once per component-set change.
    """
    n_new = len(NEW_COMPONENTS)
    alg = dict(base)
    for key in _HIST_LEAVES:
        alg[key] = _padded_like(alg[key], n_new)
    for pos, name in enumerate(sorted(NEW_COMPONENTS)):
        vec = np.zeros(len(_HIST_LEAVES) + n_new, dtype=float)
        vec[len(_HIST_LEAVES) + pos] = 1.0
        alg[name] = _one_hot_entry(vec)
    return alg


# ---------------------------------------------------------------------------
#  One-call activation: patch all algebra factories in place
# ---------------------------------------------------------------------------
def _renamed_tree_entry(tanh_entry, new_name: str):
    """Entry that calls the tanh entry and renames a hardcoded 'tanh' root.

    Two hierarchy levels hardcode the component name in the Tree root
    ('tanh' / 'tanh_h1'); reusing tanh's lambda there would silently label the
    new component as tanh in the WL-kernel features. Levels whose output is
    name-independent (Tree('node', ...)) pass through unchanged.
    """
    from cosy.core.tree import Tree

    def entry(l, i, o, p1, p2, p3, p4, p5, p6, p7):
        t = tanh_entry(l, i, o, p1, p2, p3, p4, p5, p6, p7)
        if isinstance(t, Tree) and isinstance(t.root, str) and "tanh" in t.root:
            return Tree(t.root.replace("tanh", new_name), t.children)
        return t

    return entry


def activate_extensions() -> None:
    """Patch every algebra factory so ALL consumers see the new components.

    The kernel factories in damg_kernels.py call edgelist/hierarchy/histogram
    algebras internally, so extending only your own call sites is not enough.
    Call this BEFORE importing bo_runner / kernel_experiments / damg_kernels;
    already-imported damg_kernels bindings are rebound as a fallback.

    Reuse-vs-rebuild per algebra:
      pytorch_function/model  -> new SynthSine/SynthCosine entries
      operator_histogram      -> padded basis + new one-hots (feature dim grows!)
      hierarchy               -> tanh entry with renamed Tree root where needed
      edgelist/pretty/request/refinement_1/2 -> reuse tanh entry unchanged:
        their outputs embed the label object l, so Sine()/Cosine() flow through
        and stay distinguishable automatically.
    """
    import sys

    import bayesian_optimization.examples.damg_nas.damg_repo_algebras as A

    def patched(factory, extender):
        def wrapper(*args, **kwargs):
            return extender(dict(factory(*args, **kwargs)))
        wrapper.__name__ = getattr(factory, "__name__", "algebra")
        return wrapper

    def reuse_tanh(alg):
        for name in NEW_COMPONENTS:
            alg[name] = alg["tanh"]
        return alg

    def rename_root(alg):
        for name in NEW_COMPONENTS:
            alg[name] = _renamed_tree_entry(alg["tanh"], name)
        return alg

    A.pytorch_function_algebra = patched(A.pytorch_function_algebra,
                                         lambda d: extend_algebra(d, "module"))
    A.pytorch_model_algebra = patched(A.pytorch_model_algebra,
                                      lambda d: extend_algebra(d, "module"))
    A.operator_histogram_algebra = patched(A.operator_histogram_algebra,
                                           extend_histogram_algebra)
    A.hierarchy_algebra = patched(A.hierarchy_algebra, rename_root)
    for fname in ("edgelist_algebra", "pretty_term_algebra", "request_algebra",
                  "refinement_1_algebra", "refinement_2_algebra"):
        if hasattr(A, fname):
            setattr(A, fname, patched(getattr(A, fname), reuse_tanh))

    # Fallback: rebind names inside an already-imported damg_kernels module.
    dk = sys.modules.get("bayesian_optimization.examples.damg_nas.damg_kernels")
    if dk is not None:
        for fname in ("edgelist_algebra", "hierarchy_algebra",
                      "operator_histogram_algebra"):
            if hasattr(dk, fname):
                setattr(dk, fname, getattr(A, fname))