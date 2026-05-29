"""Bayesian sleep / wellbeing model.

Predicts next-day physiological recovery and sleep quality from substance
history, temporal patterns, and autoregressive features. Shares the
target-agnostic Bayesian training core in ``work.py`` — this module only
adds the wellbeing-specific domain knowledge (which columns are valid
targets, sensible defaults, and a documented finding).

Design rationale (from predictive-qs-framework.md):
  The work model uses AW screen-time as a proxy for state because device
  physiology used to lag the export by weeks. With Whoop cycle data now
  flowing through ``load_all_df`` (recovery/HRV/RHR/strain) plus sleep
  score/duration, we can model the physiological side directly. Recovery
  is the primary wellbeing target: it's a daily 0-100 score that
  integrates HRV, resting HR, and sleep, and it responds to inputs the
  agent can reason about (substances, prior strain).

Validated finding (2026-05-28, n=1514 days, 2022-2026):
  whoop:recovery — test R²=0.186, well-calibrated CIs (94% CI covers 92%).
  The dominant substance signal is next-day suppression of recovery by
  same-day cannabinoids (standardized beta ~-0.36) and accumulated
  depressants — physiologically plausible and the strongest non-AR effect.
  This is a markedly better fit than the work-consistency model (R²=0.057),
  which matches the design intuition that physiology is more predictable
  from substance load than discretionary screen time is.
"""

import logging

import pandas as pd

from .work import BayesianWorkResult, train_bayesian_work

logger = logging.getLogger(__name__)

# Physiological / sleep targets available from load_all_df() once Whoop and
# sleep data flow through the export. Ordered by modeling preference:
# recovery is the headline wellbeing KPI; the rest are useful secondary views.
WELLBEING_TARGETS: list[str] = [
    "whoop:recovery",
    "sleep:score",
    "sleep:duration",
    "whoop:hrv",
    "whoop:resting_hr",
    "whoop:strain",
]

DEFAULT_WELLBEING_TARGET = "whoop:recovery"


def train_sleep_model(
    df: pd.DataFrame,
    target_col: str = DEFAULT_WELLBEING_TARGET,
    test_fraction: float = 0.2,
    max_features: int = 12,
    n_samples: int = 1000,
    n_tune: int = 1000,
    top_n_substances: int = 15,
    include_screentime: bool = False,
) -> BayesianWorkResult:
    """Train a Bayesian model for a sleep/wellbeing target.

    Thin wrapper over :func:`train_bayesian_work` that validates the target
    is a known physiological column and supplies wellbeing-appropriate
    defaults. The Bayesian machinery (priors, MCMC, posterior predictive,
    credible intervals) is identical — only the target differs.

    Args:
        df: Raw DataFrame from CSV export or load_all_df(). Must contain
            ``target_col`` with real (non-null) values for enough days.
        target_col: Wellbeing column to predict (default: whoop:recovery).
        test_fraction: Fraction held out for testing.
        max_features: Max features to include (keeps MCMC tractable).
        n_samples: Posterior samples per chain.
        n_tune: Tuning steps.
        top_n_substances: Number of top substances for feature building.
        include_screentime: Include AW screen-time features. AW data starts
            2021-06, so screen-time features gate the valid-row span to
            2021+. Defaults to ``False`` here (unlike the work model): screen
            time is a weak physiology predictor, and excluding it uses the
            target's full history (sleep back to 2014, substances to 2017),
            which both improves the fit and enables a genuine pre-2022
            holdout. Set ``True`` only to compare against the work model on
            the shared 2021+ span.

    Returns:
        BayesianWorkResult with trace, metrics, and posterior predictions.

    Raises:
        ValueError: If target_col is not a known wellbeing target.
        KeyError: If target_col is not present in df.
    """
    if target_col not in WELLBEING_TARGETS:
        raise ValueError(
            f"Unknown wellbeing target: {target_col!r}. "
            f"Expected one of {WELLBEING_TARGETS}"
        )
    if target_col not in df.columns:
        raise KeyError(
            f"Target {target_col!r} not in DataFrame. Whoop/sleep data may "
            f"not have flowed through the export yet — re-run qs-export.py "
            f"and confirm load_all_df() includes the column."
        )

    return train_bayesian_work(
        df,
        target_col=target_col,
        test_fraction=test_fraction,
        max_features=max_features,
        n_samples=n_samples,
        n_tune=n_tune,
        top_n_substances=top_n_substances,
        include_screentime=include_screentime,
    )
