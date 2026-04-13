"""CLI for the predictive QS framework.

Usage:
    python -m quantifiedme.predict baseline data.csv
    python -m quantifiedme.predict baseline data.csv --target time:Programming
    python -m quantifiedme.predict diagnostic data.csv
    python -m quantifiedme.predict features data.csv
    python -m quantifiedme.predict bayesian data.csv
    python -m quantifiedme.predict bayesian data.csv --target time:Programming --samples 2000
"""

import argparse
import sys
from pathlib import Path


def cmd_baseline(args: argparse.Namespace) -> None:
    """Train and evaluate a single-target baseline model."""
    from .baseline import run_baseline

    run_baseline(args.csv, target_col=args.target)


def cmd_diagnostic(args: argparse.Namespace) -> None:
    """Run multi-target diagnostic comparison."""
    from .baseline import run_diagnostic

    targets = args.targets.split(",") if args.targets else None
    run_diagnostic(args.csv, targets=targets)


def cmd_bayesian(args: argparse.Namespace) -> None:
    """Train and evaluate a Bayesian work consistency model."""
    from .baseline import load_csv_export
    from .models.work import train_bayesian_work

    df = load_csv_export(args.csv)
    result = train_bayesian_work(
        df,
        target_col=args.target,
        n_samples=args.samples,
        n_tune=args.tune,
        max_features=args.max_features,
    )
    print(result.summary())
    print()

    # Show credible intervals for recent test predictions
    ci = result.credible_intervals()
    print("Recent test predictions (last 10 days):")
    print(f"{'Date':<12} {'Actual':>8} {'Mean':>8} {'CI_3%':>8} {'CI_97%':>8} {'Hit':>5}")
    for idx, row in ci.tail(10).iterrows():
        hit = "✓" if row["ci_3"] <= row["actual"] <= row["ci_97"] else "✗"
        print(
            f"{str(idx):<12} {row['actual']:>8.2f} {row['mean']:>8.2f} "
            f"{row['ci_3']:>8.2f} {row['ci_97']:>8.2f} {hit:>5}"
        )

    # Coverage statistics
    in_94 = ((ci["actual"] >= ci["ci_3"]) & (ci["actual"] <= ci["ci_97"])).mean()
    in_50 = ((ci["actual"] >= ci["ci_25"]) & (ci["actual"] <= ci["ci_75"])).mean()
    print(f"\n94% CI coverage: {in_94:.1%} (expected: 94%)")
    print(f"50% CI coverage: {in_50:.1%} (expected: 50%)")


def cmd_features(args: argparse.Namespace) -> None:
    """Inspect feature frame: show columns, shapes, and basic stats."""
    from .baseline import load_csv_export
    from .features import build_feature_frame

    df = load_csv_export(args.csv)
    X, y = build_feature_frame(df, target_col=args.target)

    print(f"Data: {len(df)} days, {len(df.columns)} raw columns")
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} valid rows")
    print(f"Target: {args.target} (mean={y.mean():.3f}, std={y.std():.3f})")
    print()

    # Group features by prefix
    groups: dict[str, list[str]] = {}
    for col in X.columns:
        prefix = col.split(":")[0]
        groups.setdefault(prefix, []).append(col)

    print("Feature groups:")
    for prefix, cols in sorted(groups.items()):
        print(f"  {prefix}: {len(cols)} features")
        if args.verbose:
            for col in cols:
                print(f"    {col}: mean={X[col].mean():.3f}, std={X[col].std():.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantifiedme.predict",
        description="Predictive QS framework — intervention simulator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # baseline
    p_base = sub.add_parser("baseline", help="Train single-target baseline")
    p_base.add_argument("csv", type=Path, help="Path to QS CSV export")
    p_base.add_argument("--target", default="time:Work", help="Target column")
    p_base.set_defaults(func=cmd_baseline)

    # diagnostic
    p_diag = sub.add_parser("diagnostic", help="Multi-target diagnostic")
    p_diag.add_argument("csv", type=Path, help="Path to QS CSV export")
    p_diag.add_argument("--targets", default=None, help="Comma-separated target columns")
    p_diag.set_defaults(func=cmd_diagnostic)

    # bayesian
    p_bayes = sub.add_parser("bayesian", help="Train Bayesian work consistency model")
    p_bayes.add_argument("csv", type=Path, help="Path to QS CSV export")
    p_bayes.add_argument("--target", default="time:Work", help="Target column")
    p_bayes.add_argument("--samples", type=int, default=1000, help="Posterior samples per chain")
    p_bayes.add_argument("--tune", type=int, default=1000, help="Tuning steps")
    p_bayes.add_argument("--max-features", type=int, default=12, help="Max features to select")
    p_bayes.set_defaults(func=cmd_bayesian)

    # features
    p_feat = sub.add_parser("features", help="Inspect feature frame")
    p_feat.add_argument("csv", type=Path, help="Path to QS CSV export")
    p_feat.add_argument("--target", default="time:Work", help="Target column")
    p_feat.add_argument("-v", "--verbose", action="store_true", help="Show per-feature stats")
    p_feat.set_defaults(func=cmd_features)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
