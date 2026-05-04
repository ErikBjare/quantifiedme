"""Counterfactual simulation for QS predictions.

Given a trained Bayesian model, simulate "what if?" scenarios by
modifying substance features and comparing posterior predictive
distributions.

Usage (via CLI):
    python -m quantifiedme.predict simulate data.csv --add caffeine
    python -m quantifiedme.predict simulate data.csv --remove alcohol --add nicotine
    python -m quantifiedme.predict simulate data.csv --add caffeine --target time:Programming

Design:
    Interventions operate on substance features (decay:*). Adding a
    substance sets its "today" feature to 1 and adjusts kernel/count
    features accordingly. Removing sets them to 0 and reduces the
    kernel contribution.

    The simulation uses the LAST day in the dataset as the baseline
    state, so results answer: "given my recent history, what would
    tomorrow look like if I [add/remove] substance X?"
"""

import logging

import numpy as np
import pandas as pd

from .features import SUBSTANCE_DECAY_DAYS, DEFAULT_DECAY_DAYS, build_feature_frame
from .models.work import (
    BayesianWorkResult,
    query_intervention,
    select_features,
    train_bayesian_work,
)

logger = logging.getLogger(__name__)


def _find_substance_features(
    feature_names: list[str], substance: str
) -> dict[str, int]:
    """Find indices of features related to a substance.

    Returns dict mapping feature suffix ('today', 'kernel', 'count_7d')
    to the index in feature_names.
    """
    indices: dict[str, int] = {}
    prefix = f"decay:{substance}:"
    for i, name in enumerate(feature_names):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            indices[suffix] = i
    return indices


def build_intervention_features(
    X_baseline: np.ndarray,
    feature_names: list[str],
    add_substances: list[str] | None = None,
    remove_substances: list[str] | None = None,
    X_mean: pd.Series | None = None,
    X_std: pd.Series | None = None,
) -> np.ndarray:
    """Modify a standardized feature vector to reflect interventions.

    Args:
        X_baseline: Standardized feature vector (1D) for the baseline day.
        feature_names: Names of features (matching model's selected features).
        add_substances: Substances to add (set today=1, boost kernel/count).
        remove_substances: Substances to remove (set today=0, reduce kernel/count).
        X_mean: Feature means used for standardization.
        X_std: Feature stds used for standardization.

    Returns:
        Modified standardized feature vector.
    """
    X_mod = X_baseline.copy()
    add_substances = add_substances or []
    remove_substances = remove_substances or []

    for substance in add_substances:
        indices = _find_substance_features(feature_names, substance)
        if not indices:
            logger.warning(
                f"Substance '{substance}' has no features in the model "
                f"(available: {[n for n in feature_names if n.startswith('decay:')]})"
            )
            continue

        tau = SUBSTANCE_DECAY_DAYS.get(substance, DEFAULT_DECAY_DAYS)

        for suffix, idx in indices.items():
            if X_mean is not None and X_std is not None:
                mean_val = X_mean.iloc[idx]
                std_val = X_std.iloc[idx]
                if std_val == 0:
                    continue
                if suffix == "today":
                    X_mod[idx] = (1.0 - mean_val) / std_val
                elif suffix == "kernel":
                    # Add one unit dose at t=0 to existing kernel value
                    raw_baseline = X_mod[idx] * std_val + mean_val
                    raw_modified = raw_baseline + np.exp(0 / tau)  # +1 for today
                    X_mod[idx] = (raw_modified - mean_val) / std_val
                elif suffix.startswith("count_"):
                    raw_baseline = X_mod[idx] * std_val + mean_val
                    raw_modified = raw_baseline + 1.0
                    X_mod[idx] = (raw_modified - mean_val) / std_val
            else:
                if suffix == "today":
                    X_mod[idx] = 1.0
                elif suffix == "kernel":
                    X_mod[idx] += 1.0
                elif suffix.startswith("count_"):
                    X_mod[idx] += 1.0

    for substance in remove_substances:
        indices = _find_substance_features(feature_names, substance)
        if not indices:
            logger.warning(
                f"Substance '{substance}' has no features in the model"
            )
            continue

        for suffix, idx in indices.items():
            if X_mean is not None and X_std is not None:
                mean_val = X_mean.iloc[idx]
                std_val = X_std.iloc[idx]
                if std_val == 0:
                    continue
                if suffix == "today":
                    X_mod[idx] = (0.0 - mean_val) / std_val
                elif suffix == "kernel":
                    # Remove today's contribution from kernel
                    raw_baseline = X_mod[idx] * std_val + mean_val
                    raw_modified = max(0.0, raw_baseline - 1.0)
                    X_mod[idx] = (raw_modified - mean_val) / std_val
                elif suffix.startswith("count_"):
                    raw_baseline = X_mod[idx] * std_val + mean_val
                    raw_modified = max(0.0, raw_baseline - 1.0)
                    X_mod[idx] = (raw_modified - mean_val) / std_val
            else:
                if suffix == "today":
                    X_mod[idx] = 0.0
                elif suffix == "kernel":
                    X_mod[idx] = max(0.0, X_mod[idx] - 1.0)
                elif suffix.startswith("count_"):
                    X_mod[idx] = max(0.0, X_mod[idx] - 1.0)

    return X_mod


def simulate(
    df: pd.DataFrame,
    add_substances: list[str] | None = None,
    remove_substances: list[str] | None = None,
    target_col: str = "time:Work",
    n_samples: int = 1000,
    n_tune: int = 1000,
    max_features: int = 12,
) -> dict:
    """Run a counterfactual simulation.

    Trains the Bayesian model, takes the last day's features as baseline,
    applies interventions, and compares posterior predictive distributions.

    Args:
        df: Raw DataFrame from CSV export or load_all_df().
        add_substances: Substances to add to today's state.
        remove_substances: Substances to remove from today's state.
        target_col: Prediction target column.
        n_samples: Posterior samples per chain.
        n_tune: Tuning steps.
        max_features: Max features for model.

    Returns:
        Dict with keys:
            - 'baseline': posterior samples for baseline scenario
            - 'intervention': posterior samples for modified scenario
            - 'delta': difference (intervention - baseline)
            - 'summary': human-readable summary dict
            - 'model_result': BayesianWorkResult
            - 'interventions': description of what was changed
    """
    add_substances = add_substances or []
    remove_substances = remove_substances or []

    # Train the model
    logger.info("Training Bayesian model...")
    result = train_bayesian_work(
        df,
        target_col=target_col,
        n_samples=n_samples,
        n_tune=n_tune,
        max_features=max_features,
    )

    # Rebuild features to get the last day's raw feature vector
    X, y = build_feature_frame(df, target_col=target_col)

    # Feature selection (same as training)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    selected = select_features(X_train, y_train, max_features=max_features)

    X_sel = X[selected]

    # Standardization params from training data
    X_train_sel = X_train[selected]
    X_mean = X_train_sel.mean()
    X_std = X_train_sel.std().replace(0, 1)

    # Use last available day as baseline
    last_day = X_sel.iloc[-1]
    last_date = X_sel.index[-1]
    X_baseline_z = np.asarray(((last_day - X_mean) / X_std).values)

    # Build intervention features
    X_intervention_z = build_intervention_features(
        X_baseline_z,
        feature_names=selected,
        add_substances=add_substances,
        remove_substances=remove_substances,
        X_mean=X_mean,
        X_std=X_std,
    )

    # Target standardization params
    y_train_vals = y.iloc[:split_idx]
    y_mean = float(y_train_vals.mean())
    y_std_val = float(y_train_vals.std())

    # Run comparison
    comparison = query_intervention(
        result=result,
        trace=result.trace,
        baseline_features=X_baseline_z,
        modified_features=X_intervention_z,
        y_mean=y_mean,
        y_std=y_std_val,
    )

    # Build summary
    baseline_samples = comparison["baseline"]
    intervention_samples = comparison["intervention"]
    delta_samples = comparison["delta"]

    intervention_desc = []
    if add_substances:
        intervention_desc.append(f"add {', '.join(add_substances)}")
    if remove_substances:
        intervention_desc.append(f"remove {', '.join(remove_substances)}")

    summary = {
        "date": str(last_date),
        "target": target_col,
        "interventions": " + ".join(intervention_desc),
        "baseline_mean": float(np.mean(baseline_samples)),
        "baseline_ci": (
            float(np.percentile(baseline_samples, 3)),
            float(np.percentile(baseline_samples, 97)),
        ),
        "intervention_mean": float(np.mean(intervention_samples)),
        "intervention_ci": (
            float(np.percentile(intervention_samples, 3)),
            float(np.percentile(intervention_samples, 97)),
        ),
        "delta_mean": float(np.mean(delta_samples)),
        "delta_ci": (
            float(np.percentile(delta_samples, 3)),
            float(np.percentile(delta_samples, 97)),
        ),
        "prob_positive": float(np.mean(delta_samples > 0)),
    }

    return {
        "baseline": baseline_samples,
        "intervention": intervention_samples,
        "delta": delta_samples,
        "summary": summary,
        "model_result": result,
        "interventions": " + ".join(intervention_desc),
    }


def format_simulation_report(sim_result: dict) -> str:
    """Format simulation results as a human-readable report."""
    s = sim_result["summary"]
    lines = [
        f"Counterfactual Simulation: {s['target']}",
        f"  Based on: {s['date']} (most recent data)",
        f"  Intervention: {s['interventions']}",
        "",
        f"  Baseline (no change):",
        f"    Mean: {s['baseline_mean']:.2f}h",
        f"    94% CI: [{s['baseline_ci'][0]:.2f}, {s['baseline_ci'][1]:.2f}]h",
        "",
        f"  With intervention:",
        f"    Mean: {s['intervention_mean']:.2f}h",
        f"    94% CI: [{s['intervention_ci'][0]:.2f}, {s['intervention_ci'][1]:.2f}]h",
        "",
        f"  Effect (delta):",
        f"    Mean: {s['delta_mean']:+.2f}h",
        f"    94% CI: [{s['delta_ci'][0]:+.2f}, {s['delta_ci'][1]:+.2f}]h",
        f"    P(positive effect): {s['prob_positive']:.1%}",
    ]

    # Interpret the result
    if abs(s["delta_mean"]) < 0.1:
        lines.append(f"\n  Interpretation: Negligible effect (< 0.1h)")
    elif s["prob_positive"] > 0.8:
        lines.append(f"\n  Interpretation: Likely positive effect on {s['target']}")
    elif s["prob_positive"] < 0.2:
        lines.append(f"\n  Interpretation: Likely negative effect on {s['target']}")
    else:
        lines.append(f"\n  Interpretation: Uncertain effect — wide posterior overlap")

    return "\n".join(lines)
