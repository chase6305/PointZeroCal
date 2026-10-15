"""Candidate rankings compared with independent full matrix rebuilds."""

import json
from dataclasses import replace

import numpy as np
import pytest

from point_zero_cal import ContactGroup, JointZeroCalibrator, KnownDistance
from point_zero_cal.sampling import (
    CandidateRanking,
    CandidateScore,
    ObservabilityReport,
)


def observable_report():
    return ObservabilityReport(("a", "b"), 2, np.array([2.0, 1.0]), 1.0, 2.0, 6)


def test_numpy_scalar_reports_serialize_as_strict_json():
    report = replace(
        observable_report(),
        rank=np.int64(2),
        matrix_rows=np.int64(6),
        condition_number=np.float32(2),
        smallest_observable_singular_value=np.float32(1),
    )
    score = CandidateScore(np.int64(0), [0, 0], report, np.int64(0))
    payload = json.loads(
        json.dumps(CandidateRanking("g", report, (score,)).to_dict(), allow_nan=False)
    )
    assert payload["baseline"]["fully_observable"] is True
    assert payload["candidates"][0]["rank_gain"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"rank": 1.5},
        {"rank": True},
        {"rank": 3},
        {"matrix_rows": 1.5},
        {"matrix_rows": True},
        {"singular_values": [1.0]},
        {"singular_values": [1.0, 2.0]},
        {"singular_values": [2.0, 0.0]},
        {"smallest_observable_singular_value": float("nan")},
        {"smallest_observable_singular_value": 2.0},
        {"condition_number": float("nan")},
        {"condition_number": 1.0},
        {"active_joint_names": "ab"},
        {"active_joint_names": None},
    ],
)
def test_inconsistent_observability_reports_are_rejected(change):
    with pytest.raises(ValueError):
        replace(observable_report(), **change)


def test_rank_deficient_report_preserves_null_json_condition():
    report = ObservabilityReport(("a", "b"), 1, np.array([2.0, 0]), 2, np.inf, 6)
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["condition_number"] is None
    assert not payload["fully_observable"]
    with pytest.raises(ValueError):
        replace(report, condition_number=1)


def test_external_rankings_sort_candidates_and_keep_source_ties():
    weak = observable_report()
    strong = replace(weak, singular_values=[4, 2], smallest_observable_singular_value=2)
    candidates = (
        CandidateScore(0, [0, 0], weak, 0),
        CandidateScore(2, [0.2, 0], strong, 0),
        CandidateScore(1, [0.1, 0], strong, 0),
    )
    ranking = CandidateRanking("g", weak, candidates)
    assert [score.source_index for score in ranking.candidates] == [1, 2, 0]
    assert ranking.recommended.source_index == 1
    assert candidates[0].source_index == 0
    with pytest.raises(ValueError, match="baseline"):
        CandidateRanking("g", weak, (replace(candidates[0], rank_gain=1),))
    with pytest.raises(ValueError, match="baseline"):
        CandidateRanking("g", replace(weak, active_joint_names=("b", "a")), candidates)


@pytest.mark.parametrize(
    "change", [{"rank_gain": True}, {"rank_gain": 1.5}, {"configuration_rad": []}]
)
def test_invalid_candidate_scores_are_rejected(change):
    with pytest.raises(ValueError):
        replace(CandidateScore(0, [0, 0], observable_report(), 0), **change)


class AnalyticalBackend:
    dof = 3
    joint_names = ("reference", "first", "second")

    def position_and_jacobian(self, q):
        return (
            np.array([q[0] + np.sin(q[1]), np.cos(q[1]) + q[2] ** 2, np.sin(q[2])]),
            np.array(
                [
                    [1, np.cos(q[1]), 0],
                    [0, -np.sin(q[1]), 2 * q[2]],
                    [0, 0, np.cos(q[2])],
                ]
            ),
        )


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("distance_mode", ["none", "pair", "chain", "complete"])
def test_candidate_scores_match_full_rebuild(target, weighted, distance_mode):
    rng = np.random.default_rng(104)
    groups = [
        ContactGroup(
            str(index),
            rng.normal(size=(count, 3)),
            sigma_m=0.001 * (index + 1) if weighted else None,
        )
        for index, count in enumerate((2, 5, 3))
    ]
    pairs = {
        "none": [],
        "pair": [(0, 2)],
        "chain": [(0, 1), (1, 2)],
        "complete": [(0, 1), (0, 2), (1, 2)],
    }[distance_mode]
    distances = [
        KnownDistance(
            str(second),
            str(first),
            0.5,
            weight=index + 0.5,
            sigma_m=0.02 * (index + 1) if weighted else None,
        )
        for index, (first, second) in enumerate(pairs)
    ]
    candidates = rng.normal(size=(6, 3))
    candidates[-1] = candidates[0]
    offsets = np.array([0, 0.02, -0.01])
    solver = JointZeroCalibrator(AnalyticalBackend())
    ranking = solver.rank_contact_candidates(
        groups,
        candidates,
        group_name=str(target),
        distances=distances,
        offsets_rad=offsets,
    )
    active = np.array([1, 2])
    expected_scores = []
    for index, candidate in enumerate(candidates):
        augmented = list(groups)
        group = groups[target]
        augmented[target] = ContactGroup(
            group.name, np.vstack((group.samples, candidate)), sigma_m=group.sigma_m
        )
        q, plan = solver._prepare_groups(augmented, distances, "rad")
        positions, jacobians = solver._evaluate(q + offsets)
        matrix = solver._contact_information_matrix(
            tuple(positions[section] for section in plan.sections),
            tuple(jacobians[section] for section in plan.sections),
            plan,
            active,
        )
        expected = solver._observability_report(matrix, active)
        actual = next(s for s in ranking.candidates if s.source_index == index)
        single = solver.rank_contact_candidates(
            groups,
            candidate[None],
            group_name=str(target),
            distances=distances,
            offsets_rad=offsets,
        ).recommended.observability
        np.testing.assert_array_equal(single.singular_values, expected.singular_values)
        assert single.rank == expected.rank
        assert single.matrix_rows == expected.matrix_rows
        assert actual.observability.rank == expected.rank
        assert actual.observability.matrix_rows == expected.matrix_rows
        assert actual.rank_gain == expected.rank - ranking.baseline.rank
        np.testing.assert_allclose(
            actual.observability.singular_values, expected.singular_values, rtol=1e-12
        )
        assert actual.observability.condition_number == pytest.approx(
            expected.condition_number, rel=1e-12
        )
        expected_scores.append(
            (
                -expected.rank,
                -expected.smallest_observable_singular_value,
                expected.condition_number,
                index,
            )
        )
    assert [s.source_index for s in ranking.candidates] == [
        row[-1] for row in sorted(expected_scores)
    ]


@pytest.mark.parametrize("dof, counts", [(8, (2,)), (12, (2, 3)), (5, (4, 7))])
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_compressed_candidate_spectra_match_full_rows_for_wide_and_deficient_systems(
    dof, counts, rank
):
    rng = np.random.default_rng(107)
    basis = rng.normal(size=(3, dof))
    basis[:, rank:] = 0

    class Backend:
        joint_names = tuple(f"joint{k}" for k in range(dof))

        def __init__(self):
            self.dof = dof

        def position_and_jacobian(self, q):
            jacobian = basis * np.cos(q)
            return basis @ np.sin(q), jacobian

    solver = JointZeroCalibrator(Backend())
    groups = [
        ContactGroup(str(i), rng.normal(size=(count, dof)), sigma_m=0.2 * (i + 1))
        for i, count in enumerate(counts)
    ]
    candidates = rng.normal(size=(4, dof))
    ranking = solver.rank_contact_candidates(
        groups, candidates, group_name="0", fixed_joints=[]
    )
    active = np.arange(dof)
    for score in ranking.candidates:
        augmented = [
            ContactGroup(
                "0",
                np.vstack((groups[0].samples, candidates[score.source_index])),
                sigma_m=0.2,
            ),
            *groups[1:],
        ]
        q, plan = solver._prepare_groups(augmented, (), "rad")
        positions, jacobians = solver._evaluate(q)
        matrix = solver._contact_information_matrix(
            [positions[s] for s in plan.sections],
            [jacobians[s] for s in plan.sections],
            plan,
            active,
        )
        expected = solver._observability_report(matrix, active)
        actual = score.observability
        assert actual.matrix_rows == 3 * (sum(counts) + 1)
        assert actual.rank == expected.rank == rank
        assert len(actual.singular_values) == min(actual.matrix_rows, dof)
        np.testing.assert_allclose(
            actual.singular_values, expected.singular_values, rtol=1e-12, atol=1e-14
        )


def test_candidate_overflow_retries_full_centering():
    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            return np.zeros(3), np.array([[q[0] * 1e308], [0], [0]])

    solver = JointZeroCalibrator(Backend())
    groups = [ContactGroup("a", [[1], [1]])]
    with np.errstate(over="raise", invalid="raise"):
        ranking = solver.rank_contact_candidates(
            groups, [[-1]], group_name="a", fixed_joints=[]
        )
    assert ranking.baseline.rank == 0
    report = ranking.recommended.observability
    assert report.rank == 1
    assert report.matrix_rows == 9
    assert report.singular_values[0] == pytest.approx(np.sqrt(8 / 3) * 1e308 / 3)


def test_candidate_update_handles_tiny_noise_sigma():
    sigma = np.nextafter(0.0, 1.0)

    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            return np.zeros(3), np.array([[q[0] * sigma], [0], [0]])

    solver = JointZeroCalibrator(Backend())
    groups = [ContactGroup("a", [[-2], [2]], sigma_m=sigma)]
    with np.errstate(over="raise", invalid="raise"):
        ranking = solver.rank_contact_candidates(
            groups, [[0]], group_name="a", fixed_joints=[]
        )
    report = ranking.recommended.observability
    assert report.rank == 1
    assert report.singular_values[0] == pytest.approx(np.sqrt(8) / 3)


@pytest.mark.parametrize("candidate", [-1, 1])
@pytest.mark.parametrize("candidate_count", [1, 2])
def test_overflowing_qr_factor_retries_full_normalized_spectrum(
    candidate, candidate_count
):
    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            return np.array([(q[0] ** 2 - 1) * 5e307, 0, 0]), np.array(
                [[q[0] * 1e308], [0], [0]]
            )

    solver = JointZeroCalibrator(Backend())
    groups = [ContactGroup("a", [[-1], [-1], [1], [1]])]
    with np.errstate(over="raise", invalid="raise"):
        ranking = solver.rank_contact_candidates(
            groups, [[candidate]] * candidate_count, group_name="a", fixed_joints=[]
        )
    report = ranking.recommended.observability
    assert report.rank == 1
    assert report.matrix_rows == 15
    assert report.singular_values[0] == pytest.approx(np.sqrt(4.8 / 15) * 1e308)


@pytest.mark.parametrize("candidate_count", [1, 2])
def test_distance_candidate_with_coincident_centres_preserves_error(candidate_count):
    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            return np.array([q[0], 0, 0]), np.array([[1], [0], [0]])

    solver = JointZeroCalibrator(Backend())
    groups = [ContactGroup("a", [[1], [1]]), ContactGroup("b", [[-1], [-1]])]
    with pytest.raises(ValueError, match="coincident predicted centres"):
        solver.rank_contact_candidates(
            groups,
            [[-5]] * candidate_count,
            group_name="a",
            distances=[KnownDistance("a", "b", 2)],
            fixed_joints=[],
        )


def test_candidate_ranking_keeps_subnormal_contact_information():
    class Backend:
        dof = 1
        joint_names = ("joint",)

        def position_and_jacobian(self, q):
            smallest = np.nextafter(0.0, 1.0)
            return np.array([np.sin(q[0]) * smallest, 0, 0]), np.array(
                [[np.cos(q[0]) * smallest], [0], [0]]
            )

    solver = JointZeroCalibrator(Backend())
    ranking = solver.rank_contact_candidates(
        [ContactGroup("a", [[0], [np.pi]])],
        [[0], [np.pi]],
        group_name="a",
        fixed_joints=[],
    )
    assert ranking.baseline.rank == 1
    assert all(score.observability.rank == 1 for score in ranking.candidates)
    assert all(score.rank_gain == 0 for score in ranking.candidates)


def test_mixed_scale_candidate_modes_allow_an_unrepresentable_condition_number():
    class Backend:
        dof = 2
        joint_names = ("strong", "weak")

        def position_and_jacobian(self, q):
            scales = np.array([1e-14, np.nextafter(0.0, 1.0)])
            return np.r_[np.sin(q) * scales, 0], np.vstack(
                (np.diag(np.cos(q) * scales), np.zeros(2))
            )

    solver = JointZeroCalibrator(Backend(), rank_tolerance=1e-310)
    with np.errstate(over="raise", invalid="raise"):
        ranking = solver.rank_contact_candidates(
            [ContactGroup("a", [[0, 0], [np.pi, np.pi]])],
            [[0, 0], [np.pi, np.pi]],
            group_name="a",
            fixed_joints=[],
        )
    assert ranking.baseline.rank == 2
    assert np.isinf(ranking.baseline.condition_number)
    assert all(score.observability.rank == 2 for score in ranking.candidates)
    assert ranking.to_dict()["baseline"]["condition_number"] is None
