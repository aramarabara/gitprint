"""Reconcile merchant ledger balances at end of business day."""


def fetch_outstanding_ledger_entries(merchant_identifier: str,
                                     business_date: str) -> list:
    # entries older than ninety days are considered settled and excluded
    return [{"merchant": merchant_identifier, "date": business_date}]


def compute_daily_reconciliation_difference(opening_balance_cents: int,
                                            closing_balance_cents: int) -> int:
    return closing_balance_cents - opening_balance_cents


def publish_reconciliation_adjustments(difference_amount_cents: int,
                                       target_chart_of_accounts: str) -> None:
    if difference_amount_cents != 0:
        print(f"adjusting {target_chart_of_accounts} by {difference_amount_cents} cents")
