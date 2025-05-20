"""
Feature Audit Script

This script performs permutation importance analysis to identify the most
important features for a given model. It can optionally prune the dataset to
keep only the top features.

Part of Phase 1 roadmap - Feature audit & pruning via permutation importance.
"""

import argparse
import os
import pandas as pd
from src.features.feature_engineering import (
    prepare_train_test_data,
    audit_features,
    prune_features,
)
from src.models.model_factory import ModelFactory
import config


def load_data(path):
    """Load CSV data with a date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def run_audit(args):
    df = load_data(args.data)

    X_train, X_test, y_train, y_test, *_ = prepare_train_test_data(
        df,
        prune_features_flag=False,
    )

    model = ModelFactory.create_model(args.model)
    model.train(X_train, y_train)

    train_imp, test_imp, top_feats = audit_features(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        n_repeats=config.FEATURE_AUDIT_N_REPEATS,
        n_top_features=args.top_n,
        random_state=config.RANDOM_STATE,
    )

    print(f"Top {len(top_feats)} features: {top_feats}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        train_imp.to_csv(os.path.join(args.output, "train_importance.csv"), index=False)
        test_imp.to_csv(os.path.join(args.output, "test_importance.csv"), index=False)
        with open(os.path.join(args.output, "top_features.txt"), "w") as f:
            for feat in top_feats:
                f.write(f"{feat}\n")
        print(f"Results saved to {args.output}")

    if args.prune:
        X_train, X_test = prune_features(X_train, X_test, top_feats)
        print(f"Pruned dataset to {len(top_feats)} features")


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature audit")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest", "xgboost"],
        default="decision_tree",
        help="Model type to train",
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=config.TOP_N_FEATURES,
        help="Number of top features to keep",
    )
    parser.add_argument(
        "--output",
        help="Directory to save audit results",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Return pruned dataset (for future use)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
