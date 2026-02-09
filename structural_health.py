#!/usr/bin/env python3
"""
structural_health.py - Structural health assertions.

Checks for concentration risk, orphan positions, stale rebalances,
and sizing violations. Used by morning.py (reporting) and autopilot.py
(pre-trade guards).

Every bug found on 2026-02-09 (split-brain, missed rebalance, oversized
position) would have been caught by these checks on day one.
"""

import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent

CONCENTRATION_LIMIT = 0.25  # No single position > 25% of account


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def check_concentration(positions: list, equity: float,
                        limit: float = CONCENTRATION_LIMIT) -> dict:
    """
    No single position > limit% of account equity.

    Args:
        positions: list of dicts with 'symbol', 'qty', 'current_price'
        equity: total account equity
        limit: max fraction per position (default 0.25)

    Returns:
        {"status": "PASS"|"PROBLEM", "violations": [...], "max_pct": float}
    """
    violations = []
    max_pct = 0.0

    for p in positions:
        value = float(p["qty"]) * p["current_price"]
        pct = value / equity if equity > 0 else 0
        max_pct = max(max_pct, pct)
        if pct > limit:
            violations.append({
                "symbol": p["symbol"],
                "value": value,
                "pct": pct,
                "limit": limit,
            })

    return {
        "status": "PROBLEM" if violations else "PASS",
        "violations": violations,
        "max_pct": max_pct,
    }


def check_attribution(positions: list) -> dict:
    """
    Every Alpaca position must have a strategy owner in one of the ledgers.

    Args:
        positions: list of dicts with 'symbol'

    Returns:
        {"status": "PASS"|"WARNING", "unattributed": [...]}
    """
    # Load all ledgers
    main_ledger = _load_json(BASE / "trades_ledger.json")
    main_syms = set((main_ledger or {}).get("positions", {}).keys())

    xs_ledger = _load_json(BASE / "ledger_xs.json")
    xs_syms = set()
    if xs_ledger:
        xs_syms = {p["symbol"] for p in xs_ledger.get("positions", [])}

    thesis_data = _load_json(BASE / "thesis_trades.json")
    thesis_syms = set()
    if thesis_data:
        thesis_syms = {t["symbol"] for t in thesis_data.get("trades", [])
                       if t.get("status", "").lower() in ("active", "open", "pending")}

    known = main_syms | xs_syms | thesis_syms
    unattributed = [p["symbol"] for p in positions if p["symbol"] not in known]

    return {
        "status": "WARNING" if unattributed else "PASS",
        "unattributed": unattributed,
        "total": len(positions),
        "attributed": len(positions) - len(unattributed),
    }


def check_xs_freshness(max_days: int = 8) -> dict:
    """
    XS must have rebalanced within max_days.

    Args:
        max_days: max days since last rebalance (8 allows for holidays)

    Returns:
        {"status": "PASS"|"WARNING", "days_since": float|None}
    """
    state = _load_json(BASE / "autopilot_xs_state.json")
    if not state or not state.get("last_rebalance"):
        return {"status": "WARNING", "days_since": None, "detail": "never rebalanced"}

    try:
        last = datetime.fromisoformat(state["last_rebalance"])
        days = (datetime.now().astimezone() - last).total_seconds() / 86400
        stale = days > max_days
        return {
            "status": "WARNING" if stale else "PASS",
            "days_since": round(days, 1),
            "last_rebalance": state["last_rebalance"],
        }
    except Exception as e:
        return {"status": "WARNING", "days_since": None, "detail": str(e)}


def check_pretrade_concentration(symbol: str, order_shares: int,
                                 order_price: float, positions: list,
                                 equity: float,
                                 limit: float = CONCENTRATION_LIMIT) -> dict:
    """
    Pre-trade guard: would this order push a position past the concentration limit?

    Args:
        symbol: symbol being bought
        order_shares: shares in proposed order
        order_price: estimated fill price
        positions: current Alpaca positions
        equity: total account equity
        limit: max fraction per position

    Returns:
        {"allowed": bool, "reason": str, "projected_pct": float}
    """
    # Current position value in this symbol
    current_value = 0.0
    for p in positions:
        if p["symbol"] == symbol:
            current_value = float(p["qty"]) * p["current_price"]
            break

    order_value = order_shares * order_price
    projected_value = current_value + order_value
    projected_pct = projected_value / equity if equity > 0 else 0

    if projected_pct > limit:
        return {
            "allowed": False,
            "reason": (f"{symbol} would be {projected_pct:.1%} of account "
                       f"(${projected_value:,.0f} / ${equity:,.0f}), "
                       f"limit is {limit:.0%}"),
            "projected_pct": projected_pct,
        }

    return {"allowed": True, "reason": "", "projected_pct": projected_pct}


def check_pretrade_cash(order_shares: int, order_price: float,
                        cash: float) -> dict:
    """
    Pre-trade guard: does the account have enough cash for this order?

    Returns:
        {"allowed": bool, "reason": str, "order_cost": float, "cash": float}
    """
    cost = order_shares * order_price
    if cost > cash:
        return {
            "allowed": False,
            "reason": f"Order cost ${cost:,.0f} exceeds cash ${cash:,.0f}",
            "order_cost": cost,
            "cash": cash,
        }
    return {"allowed": True, "reason": "", "order_cost": cost, "cash": cash}


def run_all_checks() -> list:
    """
    Run all structural health checks. Returns list of check results.

    Each result: {"name": str, "status": "PASS"|"WARNING"|"PROBLEM", "detail": str}
    """
    results = []

    try:
        from trader import Trader
        t = Trader()
        positions = t.get_positions()
        account = t.get_account()
        equity = account["portfolio_value"]
    except Exception as e:
        return [{"name": "connection", "status": "PROBLEM",
                 "detail": f"Cannot connect to Alpaca: {e}"}]

    # Concentration
    conc = check_concentration(positions, equity)
    if conc["violations"]:
        details = "; ".join(
            f"{v['symbol']} is {v['pct']:.1%} (${v['value']:,.0f} / ${equity:,.0f})"
            for v in conc["violations"]
        )
        results.append({"name": "concentration", "status": "PROBLEM", "detail": details})
    else:
        results.append({"name": "concentration", "status": "PASS",
                        "detail": f"max {conc['max_pct']:.1%} (limit {CONCENTRATION_LIMIT:.0%})"})

    # Attribution
    attr = check_attribution(positions)
    if attr["unattributed"]:
        results.append({"name": "attribution", "status": "WARNING",
                        "detail": f"{', '.join(attr['unattributed'])} unattributed"})
    else:
        results.append({"name": "attribution", "status": "PASS",
                        "detail": f"{attr['attributed']}/{attr['total']} positions attributed"})

    # XS freshness
    xs = check_xs_freshness()
    if xs["status"] == "WARNING":
        detail = xs.get("detail", f"{xs.get('days_since', '?')} days since last rebalance (limit 8)")
        results.append({"name": "xs_freshness", "status": "WARNING", "detail": detail})
    else:
        results.append({"name": "xs_freshness", "status": "PASS",
                        "detail": f"{xs['days_since']}d since last rebalance"})

    return results
