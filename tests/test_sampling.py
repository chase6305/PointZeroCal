"""Candidate rankings compared with independent full matrix rebuilds."""

import json
from dataclasses import replace

import numpy as np
import pytest

from point_zero_cal import ContactGroup, JointZeroCalibrator, KnownDistance
from point_zero_cal.sampling import CandidateRanking, CandidateScore, ObservabilityReport


def observable_report():
    return ObservabilityReport(("a", "b"), 2, np.array([2.0, 1.0]), 1.0, 2.0, 6)


def test_numpy_scalar_reports_serialize_as_strict_json():
    report = replace(
        observable_report(), rank=np.int64(2), matrix_rows=np.int64(6),
        condition_number=np.float32(2),
        smallest_observable_singular_value=np.float32(1),
    )
    score = CandidateScore(np.int64(0), [0, 0], report, np.int64(0))
    payload = json.loads(json.dumps(
        CandidateRanking("g", report, (score,)).to_dict(), allow_nan=False
    ))
    assert payload["baseline"]["fully_observable"] is True
    assert payload["candidates"][0]["rank_gain"] == 0


@pytest.mark.parametrize("change", [
    {"rank": 1.5}, {"rank": True}, {"rank": 3},
    {"matrix_rows": 1.5}, {"matrix_rows": True},
    {"singular_values": [1.0]}, {"singular_values": [1.0, 2.0]},
    {"singular_values": [2.0, 0.0]},
    {"smallest_observable_singular_value": float("nan")},
    {"smallest_observable_singular_value": 2.0},
    {"condition_number": float("nan")}, {"condition_number": 1.0},
    {"active_joint_names": "ab"}, {"active_joint_names": None},
])
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
            np.array([
                [1, np.cos(q[1]), 0],
                [0, -np.sin(q[1]), 2 * q[2]],
                [0, 0, np.cos(q[2])],
            ]),
        )


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("with_distance", [False, True])
def test_candidate_scores_match_full_rebuild(target, weighted, with_distance):
    rng = np.random.default_rng(104)
    groups = [
        ContactGroup(
            str(index), rng.normal(size=(count, 3)),
            sigma_m=0.001 * (index + 1) if weighted else None,
        )
        for index, count in enumerate((2, 5, 3))
    ]
    distances = [KnownDistance("0", "2", 0.5)] if with_distance else []
    candidates = rng.normal(size=(6, 3))
    candidates[-1] = candidates[0]
    offsets = np.array([0, 0.02, -0.01])
    solver = JointZeroCalibrator(AnalyticalBackend())
    ranking = solver.rank_contact_candidates(
        groups, candidates, group_name=str(target), distances=distances,
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
            tuple(jacobians[section] for section in plan.sections), plan, active,
        )
        expected = solver._observability_report(matrix, active)
        actual = next(s for s in ranking.candidates if s.source_index == index)
        assert actual.observability.rank == expected.rank
        assert actual.observability.matrix_rows == expected.matrix_rows
        assert actual.rank_gain == expected.rank - ranking.baseline.rank
        np.testing.assert_allclose(
            actual.observability.singular_values, expected.singular_values, rtol=1e-12
        )
        assert actual.observability.condition_number == pytest.approx(
            expected.condition_number, rel=1e-12
        )
        expected_scores.append((
            -expected.rank, -expected.smallest_observable_singular_value,
            expected.condition_number, index,
        ))
    assert [s.source_index for s in ranking.candidates] == [
        row[-1] for row in sorted(expected_scores)
    ]
