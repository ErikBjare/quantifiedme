"""Bayesian work consistency model.

Predicts next-day work hours from substance history, temporal patterns,
and autoregressive features using PyMC. Outputs posterior predictive
distributions with uncertainty intervals rather than point estimates.

Design rationale (from predictive-qs-framework.md):
  Work output is a better proxy for overall state than sleep alone —
  sleep can look great during depression, but work output drops.
  AW data is continuous and current (unlike sleep device exports).
"""

import logging
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from ..features import build_feature_frame

logger = logging.getLogger(__name__)


@dataclass
class BayesianWorkResult:
    """Results from fitting the Bayesian work model."""

    trace: az.InferenceData
    feature_names: list[str]
    target_col: str
    n_train: int
    n_test: int
    train_r2: float
    test_r2: float
    test_rmse: float
    posterior_predictive_test: np.ndarray  # shape: (samples, n_test)
    y_test: np.ndarray
    y_test_index: pd.Index

    def summary(self) -> str:
        """Print model summary with coefficient estimates."""
        lines = [
            f"Bayesian Work Model: {self.target_col}",
            f"  Train: n={self.n_train}, R²={self.train_r2:.3f}",
            f"  Test:  n={self.n_test}, R²={self.test_r2:.3f}, RMSE={self.test_rmse:.3f}",
            "",
            "  Coefficient estimates (mean ± sd):",
        ]

        summary_df = az.summary(self.trace, var_names=["beta", "intercept"])
        # Show intercept
        if "intercept" in summary_df.index:
            row = summary_df.loc["intercept"]
            lines.append(f"    intercept: {row['mean']:.3f} ± {row['sd']:.3f}")

        # Show beta coefficients with feature names
        for i, name in enumerate(self.feature_names):
            key = f"beta[{i}]"
            if key in summary_df.index:
                row = summary_df.loc[key]
                lines.append(f"    {name}: {row['mean']:.3f} ± {row['sd']:.3f}")

        return "\n".join(lines)

    def credible_intervals(self) -> pd.DataFrame:
        """Return test predictions with 50% and 94% credible intervals."""
        ppc = self.posterior_predictive_test
        return pd.DataFrame(
            {
                "actual": self.y_test,
                "mean": ppc.mean(axis=0),
                "median": np.median(ppc, axis=0),
                "ci_3": np.percentile(ppc, 3, axis=0),
                "ci_25": np.percentile(ppc, 25, axis=0),
                "ci_75": np.percentile(ppc, 75, axis=0),
                "ci_97": np.percentile(ppc, 97, axis=0),
            },
            index=self.y_test_index,
        )


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 12,
) -> list[str]:
    """Select top features using mutual information + correlation.

    Keeps the model tractable for MCMC while retaining the most
    informative predictors. Uses a hybrid score: MI rank + abs correlation.

    Args:
        X: Feature matrix.
        y: Target variable.
        max_features: Maximum features to select.

    Returns:
        List of selected feature column names.
    """
    from sklearn.feature_selection import mutual_info_regression

    # Compute mutual information
    mi_scores = mutual_info_regression(X.values, y.values, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)

    # Compute absolute correlation
    corr_scores = X.corrwith(y).abs()

    # Hybrid rank: average of MI rank and correlation rank
    mi_rank = mi_series.rank(ascending=False)
    corr_rank = corr_scores.rank(ascending=False)
    hybrid_rank = (mi_rank + corr_rank) / 2

    selected = hybrid_rank.nsmallest(max_features).index.tolist()
    logger.info(f"Selected {len(selected)} features: {selected}")
    return selected


def train_bayesian_work(
    df: pd.DataFrame,
    target_col: str = "time:Work",
    test_fraction: float = 0.2,
    max_features: int = 12,
    n_samples: int = 1000,
    n_tune: int = 1000,
    top_n_substances: int = 15,
) -> BayesianWorkResult:
    """Train Bayesian linear model for work consistency prediction.

    Uses a regularized horseshoe prior on coefficients to handle
    many potential predictors with automatic relevance determination.

    Args:
        df: Raw DataFrame from CSV export or load_all_df().
        target_col: Column to predict (default: time:Work).
        test_fraction: Fraction held out for testing.
        max_features: Max features to include (keeps MCMC tractable).
        n_samples: Number of posterior samples per chain.
        n_tune: Number of tuning steps.
        top_n_substances: Number of top substances for feature building.

    Returns:
        BayesianWorkResult with trace, metrics, and predictions.
    """
    X, y = build_feature_frame(df, target_col=target_col, top_n_substances=top_n_substances)

    # Time-based split
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Feature selection on training data only
    selected = select_features(X_train, y_train, max_features=max_features)
    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    # Standardize features (important for prior specification)
    X_mean = X_train_sel.mean()
    X_std = X_train_sel.std().replace(0, 1)  # avoid div by zero
    X_train_z = (X_train_sel - X_mean) / X_std
    X_test_z = (X_test_sel - X_mean) / X_std

    # Standardize target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_z = (y_train - y_mean) / y_std

    n_features = len(selected)
    logger.info(
        f"Building PyMC model: {n_features} features, "
        f"{len(X_train_z)} train, {len(X_test_z)} test"
    )

    # Build PyMC model with regularized priors
    with pm.Model():
        # Priors — regularized normal (mild shrinkage)
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=0.5, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear model
        mu = intercept + pm.math.dot(X_train_z.values, beta)

        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_z.values)

        # Sample posterior
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=2,
            cores=1,  # safer in automated environments
            random_seed=42,
            progressbar=True,
            return_inferencedata=True,
        )

    # Compute posterior predictive manually (more flexible for out-of-sample)
    beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
    intercept_samples = trace.posterior["intercept"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()

    # Generate predictions: mu + noise for each posterior sample
    rng = np.random.default_rng(42)
    mu_test_z = intercept_samples[:, None] + beta_samples @ X_test_z.values.T  # (n_samples, n_test)
    noise = rng.normal(0, sigma_samples[:, None], size=mu_test_z.shape)
    ppc_z = mu_test_z + noise

    # Un-standardize
    ppc_orig = ppc_z * y_std + y_mean

    # Point predictions for metrics (posterior mean of mu, no noise)
    y_pred_test = (mu_test_z.mean(axis=0)) * y_std + y_mean

    # Train predictions via posterior mean coefficients
    beta_mean = beta_samples.mean(axis=0)
    intercept_mean = intercept_samples.mean()
    y_pred_train_z = intercept_mean + X_train_z.values @ beta_mean
    y_pred_train = y_pred_train_z * y_std + y_mean

    # Compute R² and RMSE
    def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    train_r2 = r2(y_train.values, y_pred_train)
    test_r2 = r2(y_test.values, y_pred_test)
    test_rmse = float(np.sqrt(np.mean((y_test.values - y_pred_test) ** 2)))

    return BayesianWorkResult(
        trace=trace,
        feature_names=selected,
        target_col=target_col,
        n_train=len(X_train),
        n_test=len(X_test),
        train_r2=train_r2,
        test_r2=test_r2,
        test_rmse=test_rmse,
        posterior_predictive_test=ppc_orig,
        y_test=y_test.values,
        y_test_index=y_test.index,
    )


def query_intervention(
    result: BayesianWorkResult,
    trace: az.InferenceData,
    baseline_features: np.ndarray,
    modified_features: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, np.ndarray]:
    """Compare predicted outcomes between baseline and intervention.

    Args:
        result: Fitted model result.
        trace: Posterior trace.
        baseline_features: Standardized feature vector (1D) for baseline scenario.
        modified_features: Standardized feature vector (1D) for intervention scenario.
        y_mean: Target mean for un-standardizing.
        y_std: Target std for un-standardizing.

    Returns:
        Dict with 'baseline', 'intervention', and 'delta' posterior samples.
    """
    beta_samples = trace.posterior["beta"].values.reshape(-1, len(result.feature_names))
    intercept_samples = trace.posterior["intercept"].values.flatten()

    baseline_pred = (intercept_samples + beta_samples @ baseline_features) * y_std + y_mean
    modified_pred = (intercept_samples + beta_samples @ modified_features) * y_std + y_mean

    return {
        "baseline": baseline_pred,
        "intervention": modified_pred,
        "delta": modified_pred - baseline_pred,
    }
