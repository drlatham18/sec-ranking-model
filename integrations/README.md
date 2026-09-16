# SEC Analytics / Polydesk research integration

This adapter compares the existing all-FBS preseason model against public
Polydesk full-game college-football moneyline snapshots. It reads files only.
It cannot place orders or send an execution instruction. Existing private
trading tools are not published by this integration.

## Verified September 15, 2026

Installed and verified in the existing research heartbeat on Bubbles Core under
C:/Bubbles/projects/polydesk/integrations/college-football/.
A fresh scan at 21:04 EDT retained full order books. The old scanner saved the
first five unsorted levels, which could omit the best ask; that defect is fixed.
Legacy truncated books are rejected by this adapter rather than treated as prices.

The observed North Carolina / Clemson snapshot priced Clemson at a 0.62 ask;
the August 30 preseason model estimated 0.785298, a gross difference of 16.53
percentage points. This is NOT a verified edge: the model is stale, current
season form is absent, fees and profitability are unverified, and the research
venue is global Polymarket while the configured executor targets Polymarket US.
These are different venues/contracts; prices cannot be handed from one to the
other. Snapshot data is historical, never a current trading recommendation.

## Operation

The existing run_pull.ps1 wrapper now calls run_latest.py after a successful
scan, adds a college-football/ HTML and JSON report to that same run and links it
from STATUS.md and EDGE.md. The existing schedule is unchanged.

For a single snapshot, run polydesk_compare.py with --model output/app_data.json, --run the Polydesk
run directory and --out the chosen report folder. JSON and HTML record timestamps,
model hash, missing checks and scope. Only events already present in the research
snapshot are matched. This is not a market-wide scan.

Ten regression tests cover unordered books, crossed books, team mapping, duplicate
tokens, stale models/quotes, venue mismatch, unknown teams and period contracts.
Live execution always remains disabled. No recurring task, trading permission,
account, payment, subscription or public financial recommendation is created.

## Data refresh limitation

The model covers 138 rated teams and was generated August 30, 2026. Its mtime
is not its generation time. Neither this Mac nor Bubbles Core has CFBD_API_KEY
or ~/.cfbd_key configured in the inspected execution context. Fresh source data
and a separately validated in-season model are required before interpreting this
preseason disagreement as current predictive evidence.
