"""
Adversarial Challenge Test Suite for Credit Spread Heartbeat in PositionMonitor.
Written by teamwork_preview_challenger_m1_3.

Empirically tests:
1. Live credit spread positions at 290s: exactly 0 heartbeats.
2. Live credit spread positions at 305s: exactly 1 heartbeat with milestone 1.
3. Rapid ticks at 310s, 350s, 450s: zero duplicate alerts.
4. Clock advancement to 605s: fires milestone 2.
5. Exit on Hard SL breach: exits immediately and NO heartbeat is emitted.
6. Exit on Target Decay: exits immediately and NO heartbeat is emitted.
7. Exit on Breakeven SL: exits immediately and NO heartbeat is emitted.
8. Clean teardown on remove_position().
9. Multi-position concurrency and isolation.
10. Callback exception resilience.
11. Strategy type and naming variations.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pytest

from prometheus.execution.position_monitor import PositionMonitor, TrailingState


def _make_credit_spread_state(
    position_id: str = "CS-ADV-001",
    entry_seconds_ago: int = 290,
    entry_premium: float = 40.0,
    initial_sl: float = 60.0,
    target: float = 12.0,
    strategy_type: str = "credit_spread",
    strategy: str = "credit_spread",
) -> TrailingState:
    now = datetime.now()
    entry_time_str = (now - timedelta(seconds=entry_seconds_ago)).strftime("%Y-%m-%d %H:%M:%S")
    return TrailingState(
        position_id=position_id,
        tradingsymbol="NIFTY26OCT24000CE/NIFTY26OCT24100CE",
        symbol="NIFTY",
        entry_premium=entry_premium,
        initial_sl=initial_sl,
        current_sl=initial_sl,
        target=target,
        direction="neutral_range",
        strategy=strategy,
        strategy_type=strategy_type,
        entry_time=entry_time_str,
    )


def test_credit_spread_heartbeat_milestones_and_rapid_ticks():
    """Verify 290s (0 alerts), 305s (1 alert M1), rapid ticks deduplication, and 605s (M2)."""
    heartbeat_calls = []
    exit_calls = []

    def on_heartbeat(state, current_price, elapsed_sec, milestone, health_report=None):
        heartbeat_calls.append({
            "pos_id": state.position_id,
            "current_price": current_price,
            "elapsed_sec": elapsed_sec,
            "milestone": milestone,
            "health_report": health_report,
        })

    def on_exit(pos_id, price, reason):
        exit_calls.append((pos_id, price, reason))

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=on_heartbeat,
        on_exit=on_exit,
    )

    state = _make_credit_spread_state(entry_seconds_ago=290)
    monitor.add_position(state)

    # 1. At 290s: should NOT emit any heartbeat
    monitor._process_tick(state, current_price=35.0)
    assert len(heartbeat_calls) == 0, "No heartbeat should be emitted at 290s"
    assert state._last_heartbeat_milestone == 0
    assert monitor._heartbeat_milestones.get(state.position_id) is None

    # 2. Advance clock to 305s: fires milestone 1
    now = datetime.now()
    state.entry_time = (now - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=35.0)
    assert len(heartbeat_calls) == 1, "Exactly 1 heartbeat must be emitted at 305s"
    assert heartbeat_calls[0]["milestone"] == 1
    assert heartbeat_calls[0]["elapsed_sec"] >= 300
    assert heartbeat_calls[0]["current_price"] == 35.0
    assert state._last_heartbeat_milestone == 1
    assert monitor._heartbeat_milestones[state.position_id] == 1

    # 3. Rapid ticks at 310s, 350s, 450s: zero duplicate alerts
    for elapsed in [310, 350, 450]:
        state.entry_time = (now - timedelta(seconds=elapsed)).strftime("%Y-%m-%d %H:%M:%S")
        monitor._process_tick(state, current_price=34.0)
        assert len(heartbeat_calls) == 1, f"No duplicate alerts allowed at {elapsed}s"
        assert state._last_heartbeat_milestone == 1

    # 4. Advance clock to 605s: fires milestone 2
    state.entry_time = (now - timedelta(seconds=605)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=25.0)
    assert len(heartbeat_calls) == 2, "Milestone 2 must be emitted at 605s"
    assert heartbeat_calls[1]["milestone"] == 2
    assert heartbeat_calls[1]["elapsed_sec"] >= 600
    assert heartbeat_calls[1]["current_price"] == 25.0
    assert state._last_heartbeat_milestone == 2
    assert monitor._heartbeat_milestones[state.position_id] == 2

    # 5. Rapid ticks at 610s, 700s, 890s: zero duplicate alerts
    for elapsed in [610, 700, 890]:
        state.entry_time = (now - timedelta(seconds=elapsed)).strftime("%Y-%m-%d %H:%M:%S")
        monitor._process_tick(state, current_price=24.0)
        assert len(heartbeat_calls) == 2, f"No duplicate alerts allowed at {elapsed}s"

    assert len(exit_calls) == 0, "No exit should have occurred during normal ticks"


def test_credit_spread_hard_sl_breach_no_heartbeat():
    """When credit spread breaches hard SL, exits immediately and emits NO heartbeat."""
    heartbeat_calls = []
    exit_calls = []

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a, **k: heartbeat_calls.append(a),
        on_exit=lambda *a: exit_calls.append(a),
    )

    # Position at 305s (would qualify for milestone 1 if not exiting)
    state = _make_credit_spread_state(
        entry_seconds_ago=305,
        entry_premium=40.0,
        initial_sl=60.0,  # Hard SL is 60.0
    )
    monitor.add_position(state)

    # Price jumps to 62.0 (>= 60.0) -> breaches SL
    monitor._process_tick(state, current_price=62.0)

    # Must exit immediately
    assert len(exit_calls) == 1, "Exit callback must be invoked"
    pos_id, price, reason = exit_calls[0]
    assert pos_id == "CS-ADV-001"
    assert price == 62.0
    assert reason == "stop_loss_credit_spread"

    # NO heartbeat should have been emitted
    assert len(heartbeat_calls) == 0, "No heartbeat must be emitted on SL exit tick"
    assert state._last_heartbeat_milestone == 0


def test_credit_spread_target_decay_hit_no_heartbeat():
    """When credit spread hits target decay, exits immediately and emits NO heartbeat."""
    heartbeat_calls = []
    exit_calls = []

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a, **k: heartbeat_calls.append(a),
        on_exit=lambda *a: exit_calls.append(a),
    )

    # Position at 305s (qualifies for M1 if not exiting), target is 12.0
    state = _make_credit_spread_state(
        entry_seconds_ago=305,
        entry_premium=40.0,
        target=12.0,
    )
    monitor.add_position(state)

    # Price drops to 11.0 (<= 12.0) -> hits profit target decay
    monitor._process_tick(state, current_price=11.0)

    assert len(exit_calls) == 1
    pos_id, price, reason = exit_calls[0]
    assert pos_id == "CS-ADV-001"
    assert price == 11.0
    assert reason == "target_decay_credit_spread"

    # NO heartbeat should have been emitted
    assert len(heartbeat_calls) == 0, "No heartbeat must be emitted on target decay tick"
    assert state._last_heartbeat_milestone == 0


def test_credit_spread_breakeven_sl_hit_no_heartbeat():
    """When credit spread hits breakeven SL, exits immediately and emits NO heartbeat."""
    heartbeat_calls = []
    exit_calls = []

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a, **k: heartbeat_calls.append(a),
        on_exit=lambda *a: exit_calls.append(a),
    )

    # Position at 290s: credit=40.0, breakeven decay threshold is 20.0 (50%)
    state = _make_credit_spread_state(entry_seconds_ago=290, entry_premium=40.0)
    monitor.add_position(state)

    # Step 1: price drops to 19.0 <= 20.0 (breakeven lock activates, SL ratchets to 40 * 0.85 = 34.0)
    monitor._process_tick(state, current_price=19.0)
    assert state.breakeven_set is True
    assert state.current_sl == 34.0
    assert len(heartbeat_calls) == 0  # Still at 290s

    # Step 2: advance to 305s, but price jumps to 35.0 (>= ratcheted SL of 34.0)
    now = datetime.now()
    state.entry_time = (now - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, current_price=35.0)

    assert len(exit_calls) == 1
    pos_id, price, reason = exit_calls[0]
    assert pos_id == "CS-ADV-001"
    assert price == 35.0
    assert reason == "breakeven_exit_credit_spread"

    # NO heartbeat must be emitted
    assert len(heartbeat_calls) == 0, "No heartbeat must be emitted when breakeven SL is hit"


def test_clean_teardown_on_remove_position():
    """Verify monitor.remove_position() thoroughly clears all state."""
    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a: None,
    )
    state = _make_credit_spread_state(entry_seconds_ago=310)
    monitor.add_position(state)

    # Populate internal state dicts
    monitor._process_tick(state, current_price=35.0)
    monitor._ltp_fail_counts[state.position_id] = 2
    monitor._ltp_alert_sent[state.position_id] = True

    assert state.position_id in monitor._positions
    assert state.position_id in monitor._heartbeat_milestones
    assert state.position_id in monitor._ltp_fail_counts
    assert state.position_id in monitor._ltp_alert_sent

    # Perform removal
    monitor.remove_position(state.position_id)

    assert state.position_id not in monitor._positions
    assert state.position_id not in monitor._heartbeat_milestones
    assert state.position_id not in monitor._ltp_fail_counts
    assert state.position_id not in monitor._ltp_alert_sent

    # Idempotent removal on unknown ID must not crash
    monitor.remove_position("UNKNOWN-ID")


def test_multi_position_isolation():
    """Verify multiple positions track heartbeat cadence independently."""
    calls = []

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda state, *args: calls.append((state.position_id, args[2])),
    )

    state1 = _make_credit_spread_state(position_id="CS-P1", entry_seconds_ago=305)
    state2 = _make_credit_spread_state(position_id="CS-P2", entry_seconds_ago=150)

    monitor.add_position(state1)
    monitor.add_position(state2)

    monitor._process_tick(state1, 35.0)
    monitor._process_tick(state2, 35.0)

    # Only state1 should have fired
    assert len(calls) == 1
    assert calls[0] == ("CS-P1", 1)

    # Remove state1, advance state2 to 305s
    monitor.remove_position("CS-P1")
    now = datetime.now()
    state2.entry_time = (now - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state2, 35.0)

    assert len(calls) == 2
    assert calls[1] == ("CS-P2", 1)


def test_heartbeat_resilience_to_callback_exception():
    """Verify exceptions in on_heartbeat callback do not disrupt monitoring flow."""
    def buggy_callback(*args, **kwargs):
        raise RuntimeError("Simulated crash in telegram dispatcher")

    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=buggy_callback,
    )

    state = _make_credit_spread_state(entry_seconds_ago=305)
    monitor.add_position(state)

    # Should not raise exception
    try:
        monitor._process_tick(state, current_price=35.0)
    except Exception as e:
        pytest.fail(f"_process_tick raised an exception during failing heartbeat: {e}")

    # Milestone should still be set to avoid retry loops
    assert state._last_heartbeat_milestone == 1


def test_strategy_type_detection_variants():
    """Verify credit spread detection handles various naming schemes."""
    cases = [
        ("credit_spread", ""),
        ("", "credit_spread"),
        ("", "CREDIT_SPREAD"),
        ("", "BULL_PUT_SPREAD"),
        ("credit_spread", "Iron Condor"),
    ]

    for strat_type, strat_name in cases:
        calls = []
        monitor = PositionMonitor(
            broker=MagicMock(),
            poll_interval=1,
            on_heartbeat=lambda *a, **k: calls.append(a),
        )
        state = _make_credit_spread_state(
            position_id=f"CS-{strat_type}-{strat_name}",
            entry_seconds_ago=305,
            strategy_type=strat_type,
            strategy=strat_name,
        )
        monitor.add_position(state)
        monitor._process_tick(state, 35.0)
        assert len(calls) == 1, f"Failed for strategy_type='{strat_type}', strategy='{strat_name}'"


def test_heartbeat_callback_signatures_backward_compatibility():
    """Verify PositionMonitor accommodates 5-arg, 4-arg, and 2-arg on_heartbeat callbacks."""
    # 1. 5 arguments
    c5 = []
    m5 = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda state, price, sec, m, rep: c5.append((state.position_id, price, sec, m, rep)),
    )
    s5 = _make_credit_spread_state(position_id="CS-SIG5", entry_seconds_ago=305)
    m5.add_position(s5)
    m5._process_tick(s5, 35.0)
    assert len(c5) == 1
    assert c5[0][3] == 1

    # 2. 4 arguments
    c4 = []
    m4 = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda state, price, sec, m: c4.append((state.position_id, price, sec, m)),
    )
    s4 = _make_credit_spread_state(position_id="CS-SIG4", entry_seconds_ago=305)
    m4.add_position(s4)
    m4._process_tick(s4, 35.0)
    assert len(c4) == 1
    assert c4[0][3] == 1

    # 3. 2 arguments
    c2 = []
    m2 = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda state, price: c2.append((state.position_id, price)),
    )
    s2 = _make_credit_spread_state(position_id="CS-SIG2", entry_seconds_ago=305)
    m2.add_position(s2)
    m2._process_tick(s2, 35.0)
    assert len(c2) == 1
    assert c2[0][1] == 35.0


def test_boundary_holding_seconds():
    """Verify precise mathematical second boundaries: 299s (0), 300s (M1), 599s (M1), 600s (M2)."""
    calls = []
    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a: calls.append(a[3]),
    )
    now = datetime.now()
    state = _make_credit_spread_state(entry_seconds_ago=0)
    monitor.add_position(state)

    # 299s -> M=0, no fire
    state.entry_time = (now - timedelta(seconds=299)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 0

    # 300s -> M=1, fires
    state.entry_time = (now - timedelta(seconds=300)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 1
    assert calls[0] == 1

    # 599s -> M=1, duplicate milestone, no fire
    state.entry_time = (now - timedelta(seconds=599)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 1

    # 600s -> M=2, fires
    state.entry_time = (now - timedelta(seconds=600)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 2
    assert calls[1] == 2

    # 899s -> M=2, duplicate milestone, no fire
    state.entry_time = (now - timedelta(seconds=899)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 2

    # 900s -> M=3, fires
    state.entry_time = (now - timedelta(seconds=900)).strftime("%Y-%m-%d %H:%M:%S")
    monitor._process_tick(state, 35.0)
    assert len(calls) == 3
    assert calls[2] == 3


def test_entry_time_format_resilience():
    """Verify PositionMonitor defensively parses diverse entry_time formats without crashing."""
    formats = [
        datetime.now() - timedelta(seconds=305),
        (datetime.now() - timedelta(seconds=305)).isoformat(),
        (datetime.now() - timedelta(seconds=305)).strftime("%Y-%m-%d %H:%M:%S"),
        "",
        None,
    ]

    for fmt in formats:
        calls = []
        monitor = PositionMonitor(
            broker=MagicMock(),
            poll_interval=1,
            on_heartbeat=lambda *a: calls.append(a),
        )
        state = _make_credit_spread_state(entry_seconds_ago=10)
        state.entry_time = fmt
        monitor.add_position(state)
        # Must not raise exception
        monitor._process_tick(state, 35.0)
        if fmt in ("", None):
            assert len(calls) == 0
        else:
            assert len(calls) == 1


def test_concurrent_tick_processing_deduplication():
    """Verify rapid concurrent threads calling _process_tick produce exactly 1 alert per milestone."""
    import concurrent.futures

    calls = []
    monitor = PositionMonitor(
        broker=MagicMock(),
        poll_interval=1,
        on_heartbeat=lambda *a: calls.append(a),
    )
    state = _make_credit_spread_state(entry_seconds_ago=305)
    monitor.add_position(state)

    def worker():
        for _ in range(50):
            monitor._process_tick(state, 35.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker) for _ in range(8)]
        for f in futures:
            f.result()

    # Even with 400 concurrent ticks at 305s, milestone 1 must only fire once!
    assert len(calls) == 1
    assert state._last_heartbeat_milestone == 1

