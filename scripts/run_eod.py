"""End-of-day risk batch: run once, or stay running and execute on a daily schedule.

    python scripts/run_eod.py                         # run now for the configured portfolio
    python scripts/run_eod.py --all                   # every portfolio in the database
    python scripts/run_eod.py --portfolio GMF001 --reports-dir reports
    python scripts/run_eod.py --schedule 18:00        # run every weekday at 18:00 (local time), forever

The job initialises the database if needed (schema, sample data, scenario
library), then computes and persists VaR / ES for every methodology, risk
contributions, the stress library, P&L attribution and VaR backtests, and
writes the Excel and PDF reports. It works on SQLite and on PostgreSQL
(``DATABASE_URL=postgresql+psycopg://...``).

Exit codes: 0 success, 1 a run failed, 2 success with risk-limit breaches
(only when ``--fail-on-breach`` is set), so a scheduler or CI job can alert.

For production scheduling a cron entry or orchestrator is equivalent to the
built-in loop, e.g. ``0 18 * * 1-5 cd /app && python scripts/run_eod.py``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import configure_logging, get_settings  # noqa: E402
from services.bootstrap_service import initialise_database  # noqa: E402
from services.eod_service import EodResult, run_end_of_day  # noqa: E402

logger = logging.getLogger("eod")
_stop = False


def _handle_signal(signum, _frame) -> None:  # pragma: no cover - signal handler
    global _stop
    _stop = True
    logger.info("Signal %s received; stopping after the current run", signum)


def parse_time(value: str) -> tuple[int, int]:
    """Parse ``HH:MM`` (24h) into (hour, minute)."""
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("schedule must be HH:MM (24-hour), e.g. 18:00") from exc
    return parsed.hour, parsed.minute


def next_run_time(now: datetime, hour: int, minute: int, weekdays_only: bool = True) -> datetime:
    """Next occurrence of ``hour:minute`` strictly after ``now`` (skipping weekends if asked)."""
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while weekdays_only and candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def run_once(portfolio_ids: list[str] | None, all_portfolios: bool, reports_dir: Path | None) -> list[EodResult]:
    """Initialise the database and run the batch for the requested portfolios."""
    settings = get_settings()
    repo, _ = initialise_database(settings)
    if all_portfolios:
        ids = repo.list_portfolios()["portfolio_id"].tolist()
    else:
        ids = portfolio_ids or [settings.portfolio.portfolio_id]
    results = []
    for pid in ids:
        results.append(run_end_of_day(repo, settings, pid, reports_dir))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-of-day risk batch.")
    parser.add_argument("--portfolio", action="append", help="Portfolio ID (repeatable). Default: configured portfolio.")
    parser.add_argument("--all", action="store_true", help="Run every portfolio in the database.")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports", help="Where to write Excel / PDF reports.")
    parser.add_argument("--no-reports", action="store_true", help="Skip report generation.")
    parser.add_argument("--schedule", type=parse_time, metavar="HH:MM", help="Stay running and execute daily at this local time.")
    parser.add_argument("--include-weekends", action="store_true", help="With --schedule: also run on Saturday and Sunday.")
    parser.add_argument("--fail-on-breach", action="store_true", help="Exit with code 2 when a risk limit is breached.")
    args = parser.parse_args(argv)

    configure_logging(get_settings())
    reports_dir = None if args.no_reports else args.reports_dir

    def execute() -> int:
        try:
            results = run_once(args.portfolio, args.all, reports_dir)
        except Exception:  # noqa: BLE001 - report and signal failure to the scheduler
            logger.exception("EOD run failed")
            return 1
        for r in results:
            print(
                f"{r.portfolio_id} {r.valuation_date}: NAV {r.nav:,.0f} | VaR99 {r.var_99:,.0f} | ES99 {r.es_99:,.0f} | "
                f"worst stress {r.worst_stress} {r.worst_stress_pnl:,.0f} | breaches {len(r.limit_breaches)} | {r.duration_seconds}s"
            )
        breached = any(r.limit_breaches for r in results)
        return 2 if (breached and args.fail_on_breach) else 0

    if args.schedule is None:
        return execute()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    hour, minute = args.schedule
    logger.info("EOD scheduler started: daily at %02d:%02d%s", hour, minute, "" if args.include_weekends else " (weekdays)")
    while not _stop:
        due = next_run_time(datetime.now(), hour, minute, not args.include_weekends)
        logger.info("Next EOD run at %s", due.isoformat(timespec="minutes"))
        while not _stop and datetime.now() < due:
            time.sleep(min(30.0, max((due - datetime.now()).total_seconds(), 0.0)))
        if not _stop:
            execute()
    return 0


if __name__ == "__main__":
    sys.exit(main())
