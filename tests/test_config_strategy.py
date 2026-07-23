"""
Guard config.PRODUCTION_WEIGHT_STRATEGY against drift.

PRODUCTION_WEIGHT_STRATEGY is the single constant consumed by BOTH
fpl.model.predict and fpl.run_week to decide how per-position model members
are combined (see fpl/config.py's docstring and fpl/model/train.py's
combination bake-off). If it is ever edited to a typo'd model name or an
unsupported strategy string, that breaks BOTH the backtest and the live
weekly run identically and silently (no test currently catches it) - so this
guards the string's shape at import time rather than only when a pipeline
run happens to hit it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl import config
from fpl.model.ensemble import _WEIGHT_FITTERS
from fpl.model.models import FACTORIES

# Non-"single:" strategy names the combination bake-off actually tries, mirrored from
# fpl/model/train.py::evaluate_static_split's `candidates` list (single:catboost aside,
# which is handled separately below). These are also exactly the keys of
# fpl.model.ensemble._WEIGHT_FITTERS, which fit_weights() dispatches on - so we assert
# equality with that dict below rather than trusting this comment to stay in sync.
NON_SINGLE_STRATEGIES = {"nnls", "top_k", "ridge"}


def test_production_weight_strategy_is_a_nonempty_string():
    assert isinstance(config.PRODUCTION_WEIGHT_STRATEGY, str)
    assert config.PRODUCTION_WEIGHT_STRATEGY.strip() != ""


def test_single_strategy_model_name_is_a_known_factory():
    strategy = config.PRODUCTION_WEIGHT_STRATEGY
    if strategy.startswith("single:"):
        model_name = strategy.split(":", 1)[1]
        assert model_name in FACTORIES, (
            f"config.PRODUCTION_WEIGHT_STRATEGY={strategy!r} names a model not in "
            f"fpl.model.models.FACTORIES ({sorted(FACTORIES)})"
        )


def test_non_single_strategy_is_a_recognized_combiner():
    strategy = config.PRODUCTION_WEIGHT_STRATEGY
    if not strategy.startswith("single:"):
        # Source of truth: fpl.model.ensemble.fit_weights dispatches on this exact dict.
        assert strategy in _WEIGHT_FITTERS
        assert set(_WEIGHT_FITTERS.keys()) == NON_SINGLE_STRATEGIES
