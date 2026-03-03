import numpy as np

from src.strategies.tris.layered_mesh_scorer import LayeredMeshScorer


def _uniform_pmf() -> np.ndarray:
    return np.full((5, 10), 0.1, dtype=np.float64)


def test_layered_mesh_scorer_output_shapes_and_component_keys():
    scorer = LayeredMeshScorer()
    tickets = np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]], dtype=np.int16)
    out = scorer.score_all(tickets=tickets, pmf_pos=_uniform_pmf())

    assert isinstance(out, dict)
    assert set(out.keys()) == {"total_score", "components"}
    assert np.asarray(out["total_score"]).shape == (2,)
    components = out["components"]
    assert set(components.keys()) == {
        "positional_logp",
        "hamming_memory",
        "cross_turbulence",
        "camera_repeat_penalty",
    }
    assert all(np.asarray(components[k]).shape == (2,) for k in components)


def test_camera_repeat_penalty_is_soft_and_position_weighted():
    scorer = LayeredMeshScorer(
        weights={
            "camera_repeat_penalty_per_pos": [1.0, 2.0, 0.0, 0.0, 0.0],
            "positional_logp": 0.0,
            "hamming_memory": 0.0,
            "cross_turbulence": 0.0,
            "camera_repeat_penalty": 1.0,
        }
    )
    prev = [1, 2, 3, 4, 5]
    tickets = np.array(
        [
            [1, 2, 8, 8, 8],  # repite pos0+pos1 => -3
            [1, 9, 8, 8, 8],  # repite solo pos0 => -1
            [9, 9, 8, 8, 8],  # no repite => 0
        ],
        dtype=np.int16,
    )
    out = scorer.score_all(tickets=tickets, pmf_pos=_uniform_pmf(), prev_digits=prev)
    rep = np.asarray(out["components"]["camera_repeat_penalty"], dtype=np.float64)

    np.testing.assert_allclose(rep, np.array([-3.0, -1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(out["total_score"], rep)


def test_hamming_memory_returns_zero_without_empirical_context():
    scorer = LayeredMeshScorer(
        weights={
            "positional_logp": 0.0,
            "hamming_memory": 1.0,
            "cross_turbulence": 0.0,
            "camera_repeat_penalty": 0.0,
        }
    )
    tickets = np.array([[0, 0, 0, 0, 0], [9, 9, 9, 9, 9]], dtype=np.int16)
    out = scorer.score_all(
        tickets=tickets,
        pmf_pos=_uniform_pmf(),
        prev_digits=[1, 1, 1, 1, 1],
        camera_diag=None,
    )
    ham = np.asarray(out["components"]["hamming_memory"], dtype=np.float64)
    np.testing.assert_allclose(ham, np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(out["total_score"], ham)


def test_hamming_memory_uses_empirical_distribution_when_available():
    scorer = LayeredMeshScorer(
        weights={
            "positional_logp": 0.0,
            "hamming_memory": 1.0,
            "cross_turbulence": 0.0,
            "camera_repeat_penalty": 0.0,
            "hamming_min_history": 5,
        }
    )
    prev = [1, 1, 1, 1, 1]
    tickets = np.array(
        [
            [1, 1, 1, 1, 1],  # D=0
            [9, 9, 9, 9, 9],  # D=5
        ],
        dtype=np.int16,
    )
    # Empirico favorece distancia 0 y castiga distancia 5.
    diag = {"hamming_hist_counts": [50, 5, 2, 1, 1, 1]}
    out = scorer.score_all(
        tickets=tickets,
        pmf_pos=_uniform_pmf(),
        prev_digits=prev,
        camera_diag=diag,
    )
    ham = np.asarray(out["components"]["hamming_memory"], dtype=np.float64)
    assert float(ham[0]) > float(ham[1])
