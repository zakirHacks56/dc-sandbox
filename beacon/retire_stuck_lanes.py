"""One-off: retire today's stuck IMPLEMENTING lanes so the hunter unblocks.

Background: 2026-09-28's fixer runs all exited 0 in 1.5-4 min without a PR.
Every run died on PreflightTokenBudgetExceeded (calls needed 41k-94k input
tokens vs the 40k routine budget) or a parse/regression-test rejection, and
the budget-exceeded path `return`ed without writing a terminal record state.
The board lanes stayed IMPLEMENTING, each repo ate its whole 4-lane/day
budget, and the hunter reported "no candidate found this tick" for hours.

This marks those lanes ABANDONED with their ORIGINAL updated timestamp so
they drop out of today's lane count (exactly what _sweep_stale does), and
stamps spent=0 -- preflight refusals spend no tokens, so under the new
lane-accounting rule they must not burn the daily cap.

Run from the repo root:  python beacon/retire_stuck_lanes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beacon_util as util  # noqa: E402

INFLIGHT_STATES = ("ANALYZING", "IMPLEMENTING", "WAITING_FOR_FEEDBACK",
                   "REVIEWING", "GATED", "PAUSED")

# Exact lanes stuck from the 2026-09-28 bad-context preflight wave.
STUCK_LANES = [
    "yunaremaia/sandbox-ffi-layers#13",
    "yunaremaia/sandbox-ffi-layers#12",
    "yunaremaia/sandbox-ffi-layers#11",
    "Rekin226/aquascope#392",
    "Rekin226/aquascope#381",
    "Rekin226/aquascope#376",
    "Rekin226/aquascope#375",
    "taranis-ai/taranis-ai#626",
    "taranis-ai/taranis-ai#623",
    "taranis-ai/taranis-ai#556",
    "taranis-ai/taranis-ai#553",
    "izzywdev/FuzeFront#1016",
    "izzywdev/FuzeFront#1015",
    "izzywdev/FuzeFront#1003",
    "izzywdev/FuzeFront#1002",
]


def maybe_retire(board: dict) -> int:
    lanes = board.setdefault("lanes", {})
    retired = 0
    for lane_key, lane in lanes.items():
        if lane_key not in STUCK_LANES:
            continue
        state = str(lane.get("state", ""))
        if state not in INFLIGHT_STATES:
            continue
        if lane.get("pr"):
            continue  # never touch lanes with a real PR
        lanes[lane_key] = {
            "state": "ABANDONED",
            "decision": False,
            "spent": 0,
            "reason": "one-off retire: budget-exceeded/preflight wave left no "
                      "terminal state on 2026-09-28",
            "updated": lane.get("updated")
                       or lane.get("updated_at")
                       or util.now_utc(),
        }
        retired += 1
    return retired


def main() -> int:
    path = Path(util.BOARD)
    board = util.load_json(path, {"lanes": {}})
    retired = maybe_retire(board)
    print(f"retired {retired} stuck lane(s)")
    if not retired:
        return 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(board, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())