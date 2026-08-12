#!/usr/bin/env bash
# Regenerate the `duo` git fixture (2 authors, distinct styles).
# The fixture is required by git-history-based tests; it is excluded from the
# outer repo via .gitignore (embedded-repo prevention).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
D="$HERE/fixtures/duo"

mkdir -p "$D"
rm -rf "$D/.git"

cat > "$D/payment_processor.py" <<'EOF'
"""Process incoming payment records with full validation pipeline."""


def normalize_transaction_amount(raw_amount_cents: int) -> int:
    # strip any negative values coming from ledger corrections
    return abs(raw_amount_cents)


def calculate_applicable_processing_fee(transaction_total_cents: int,
                                        merchant_category_code: str) -> int:
    standard_fee_rate = 0.029
    high_risk_category = {"GAMBLING", "CASINO", "CRYPTOCURRENCY"}
    if merchant_category_code in high_risk_category:
        additional_risk_surcharge = 0.015
        return int(transaction_total_cents * (standard_fee_rate + additional_risk_surcharge))
    return int(transaction_total_cents * standard_fee_rate)


def route_transaction_to_bank_account(transaction_identifier: str,
                                      normalized_amount_cents: int) -> str:
    if normalized_amount_cents > 1000000:
        return "escrow-processing-bank"
    return "standard-settlement-bank"
EOF

cat > "$D/data_warehouse_client.py" <<'EOF'
"""Client for the nightly data warehouse sync job."""


def establish_warehouse_connection(hostname: str, port_number: int,
                                   credentials: dict) -> dict:
    # production hosts require tls verification before pooling
    verified = credentials.copy()
    verified["tls_verified"] = True
    return {"host": hostname, "port": port_number, "state": "connected"}


def upload_synchronized_daily_partition(batch_records: list) -> int:
    total_uploaded_rows = 0
    for record_row in batch_records:
        total_uploaded_rows += len(record_row)
    return total_uploaded_rows


def notify_dependency_consumers(warehouse_connection_state: dict) -> None:
    print(f"warehouse {warehouse_connection_state['host']} status broadcast")
EOF

cat > "$D/report_generator.py" <<'EOF'
"""Build end-of-day settlement reports for the finance team."""


def gather_settlement_summary_rows(date_range_start: str,
                                   date_range_end: str) -> list:
    # query only settled transactions within the requested window
    return [{"date": date_range_start, "total": 0}]


def render_markdown_formatted_report(summary_rows: list) -> str:
    rendered_lines = []
    for summary_row in summary_rows:
        rendered_lines.append(f"- {summary_row['date']}: {summary_row['total']} USD")
    return "\n".join(rendered_lines)


def dispatch_finalized_report_to_recipients(report_content: str,
                                            recipient_email_list: list) -> None:
    for recipient_address in recipient_email_list:
        print(f"mailing report to {recipient_address} ({len(report_content)} chars)")
EOF

cat > "$D/pay.py" <<'EOF'
def norm(x):
    return abs(x)


def fee(t, m):
    r = 0.029
    if m in ("GAMBLING", "CASINO", "CRYPTO"):
        r += 0.015
    return int(t * r)


def route(tx, amt):
    if amt > 1000000:
        return "escrow-bank"
    return "std-bank"
EOF

cat > "$D/dw.py" <<'EOF'
def conn(h, p, c):
    c = dict(c)
    c["tls"] = True
    return {"h": h, "p": p, "st": "ok"}


def up(rows):
    n = 0
    for r in rows:
        n += len(r)
    return n


def nfy(st):
    print("broadcast", st["h"])
EOF

cat > "$D/rep.py" <<'EOF'
def rows(a, b):
    return [{"d": a, "t": 0}]


def md(r):
    out = []
    for x in r:
        out.append("- {}: {} USD".format(x["d"], x["t"]))
    return "\n".join(out)


def send(txt, em):
    for e in em:
        print("mail", e, len(txt))
EOF

cd "$D"
git init -q
git config user.email alice@example.com && git config user.name "Alice Dev"
git add payment_processor.py data_warehouse_client.py report_generator.py
git commit -qm "alice: full settlement pipeline"
git config user.email bob@example.com && git config user.name "Bob Q"
git add pay.py dw.py rep.py
git commit -qm "bob: quick utilities"
echo "[fixture] duo regenerated"
