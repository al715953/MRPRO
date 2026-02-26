import numpy as np

from src.strategies.tris.structural_filters import StructuralFilterConfig, StructuralFilterEngine


def test_mask_all_legacy_mode_matches_previous_behavior():
    cfg = StructuralFilterConfig()
    tickets = np.array(
        [
            [1, 2, 3, 4, 5],  # 5 repeats
            [1, 2, 3, 4, 9],  # 4 repeats
            [1, 2, 3, 8, 9],  # 3 repeats
            [1, 2, 7, 8, 9],  # 2 repeats
            [9, 9, 9, 9, 9],  # 0 repeats
        ],
        dtype=np.uint8,
    )
    prev = [1, 2, 3, 4, 5]
    static_mask = np.array([True, True, True, True, False], dtype=bool)

    got = StructuralFilterEngine.mask_all(tickets, prev, static_mask, cfg)
    expected = static_mask & (np.sum(tickets.astype(np.int16) == np.array(prev)[None, :], axis=1) <= 2)
    np.testing.assert_array_equal(got, expected)


def test_camera_aware_disabling_global_sum_and_parity_avoids_those_reasons():
    cfg = StructuralFilterConfig(
        enable_global_sum_filter=False,
        enable_global_parity_filter=False,
        min_unique_digits=1,
        max_consecutive_run=5,
        max_positional_repeats_vs_prev=5,
        hard_filter=True,
    )
    engine = StructuralFilterEngine(cfg)
    ticket = [9, 9, 9, 9, 9]  # Sum/paridad fuera de defaults legacy.

    violations = engine._violations(ticket, None, cfg)
    assert "sum" not in violations
    assert "parity" not in violations

    accepted, diag = engine.apply([ticket], prev_digits=None)
    assert accepted == [ticket]
    assert int(diag["reject_reasons"].get("sum", 0)) == 0
    assert int(diag["reject_reasons"].get("parity", 0)) == 0


def test_positional_forbidden_digits_applies_per_position():
    cfg = StructuralFilterConfig(
        positional_limits=[
            {"forbidden_digits": [7]},
            {},
            {},
            {},
            {},
        ],
        max_positional_repeats_vs_prev=5,
    )
    tickets = np.array(
        [
            [7, 1, 1, 1, 1],
            [6, 1, 1, 1, 1],
            [7, 9, 9, 9, 9],
        ],
        dtype=np.uint8,
    )
    static_mask = np.array([True, True, True], dtype=bool)

    mask = StructuralFilterEngine.mask_all(tickets, prev_digits=None, static_mask=static_mask, cfg=cfg)
    np.testing.assert_array_equal(mask, np.array([False, True, False], dtype=bool))

    engine = StructuralFilterEngine(cfg)
    violations = engine._violations([7, 1, 1, 1, 1], None, cfg)
    assert "pos1_forbidden" in violations


def test_immediate_repeat_per_position_blocks_only_marked_positions():
    cfg = StructuralFilterConfig(
        immediate_repeat_mode="per_position",
        immediate_repeat_disallow_positions=(True, False, True, False, False),
        max_positional_repeats_vs_prev=5,
    )
    prev = [1, 2, 3, 4, 5]
    tickets = np.array(
        [
            [1, 9, 3, 9, 9],  # Repite pos1 y pos3 (bloqueado)
            [1, 9, 8, 9, 9],  # Repite solo pos1 (bloqueado)
            [8, 2, 8, 4, 5],  # Repite solo posiciones permitidas (aceptado)
            [8, 9, 8, 9, 9],  # Sin repeats prohibidos (aceptado)
        ],
        dtype=np.uint8,
    )
    static_mask = np.array([True, True, True, True], dtype=bool)

    mask = StructuralFilterEngine.mask_all(tickets, prev, static_mask, cfg)
    np.testing.assert_array_equal(mask, np.array([False, False, True, True], dtype=bool))

    engine = StructuralFilterEngine(cfg)
    violations = engine._violations([1, 9, 8, 9, 9], np.asarray(prev, dtype=np.int16), cfg)
    assert "pos1_repeat_prev" in violations
    assert "pos2_repeat_prev" not in violations
