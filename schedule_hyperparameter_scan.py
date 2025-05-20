#!/usr/bin/env python
"""Hyperparameter Scan Scheduler

Runs the hyperparameter optimization CLI at regular intervals.
"""

import argparse
import subprocess
import logging
import schedule
import time


def run_scan(data_path: str, model: str, trials: int | None, regime_specific: bool) -> None:
    """Execute the hyperparameter optimization script."""
    cmd = [
        "python",
        "optimize_hyperparameters.py",
        "--data",
        data_path,
        "--model",
        model,
    ]
    if regime_specific:
        cmd.append("--regime-specific")
    if trials is not None:
        cmd.extend(["--trials", str(trials)])

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schedule recurring hyperparameter optimization jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to data file or directory")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest", "xgboost", "all"],
        default="all",
        help="Model type to optimize",
    )
    parser.add_argument("--trials", type=int, default=None, help="Number of Optuna trials")
    parser.add_argument("--regime-specific", action="store_true", help="Enable regime specific optimization")
    parser.add_argument("--day", default="sunday", help="Day of week to run the scan")
    parser.add_argument("--time", default="02:00", help="Time of day to run the scan (HH:MM)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hyperparam_scheduler")

    def job():
        logger.info("Starting scheduled hyperparameter scan")
        try:
            run_scan(args.data, args.model, args.trials, args.regime_specific)
            logger.info("Hyperparameter scan completed")
        except subprocess.CalledProcessError as exc:
            logger.error("Hyperparameter scan failed: %s", exc)

    schedule_map = {
        "monday": schedule.every().monday,
        "tuesday": schedule.every().tuesday,
        "wednesday": schedule.every().wednesday,
        "thursday": schedule.every().thursday,
        "friday": schedule.every().friday,
        "saturday": schedule.every().saturday,
        "sunday": schedule.every().sunday,
    }
    day = args.day.lower()
    scheduler = schedule_map.get(day, schedule.every().sunday)
    scheduler.at(args.time).do(job)

    logger.info("Scheduled hyperparameter scan every %s at %s", day, args.time)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
