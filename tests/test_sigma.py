"""Measurement-noise scales (sigma_m) for contact groups and known distances."""

import json

import numpy as np
import pytest

from point_zero_cal import (
    ContactGroup,
    JointZeroCalibrator,
    KnownDistance,
)


class PlanarArm:
    """Two-link analytic arm matching the planar fixture geometry."""

    dof = 2
    joint_names = ("shoulder", "elbow")

    def position_and_jacobian(self, q):
        q1, q2 = q
        position = np.array(
            [np.cos(q1) + np.cos(q1 + q2), np.sin(q1) + np.sin(q1 + q2), 0.0]
        )
        jacobian = np.array(
            [
                [-np.sin(q1) - np.sin(q1 + q2), -np.sin(q1 + q2)],
                [np.cos(q1) + np.cos(q1 + q2), np.cos(q1 + q2)],
                [0.0, 0.0],
            ]
        )
        return position, jacobian


def noisy_groups():
    """A clean and a noisy group touching the points (1, 1) and (sqrt(3), 0)."""
    true = np.array([0.0, -0.025])
    clean_q = np.array([[0, np.pi / 2], [np.pi / 2, -np.pi / 2]])
    noisy_q = np.array([[-np.pi / 6, np.pi / 3], [np.pi / 6, -np.pi / 3]])
    noise = np.array([[0.006, -0.004], [0.002, 0.008]])
    return (
        ContactGroup("clean", clean_q - true),
        ContactGroup("noisy", noisy_q - true + noise),
        KnownDistance("clean", "noisy", float(np.hypot(np.sqrt(3) - 1, 1))),
    )


def test_contact_group_sigma_m_requires_positive_finite_metres():
    with pytest.raises(ValueError):
        ContactGroup("a", np.zeros((2, 2)), sigma_m=0)
    for value in (-1, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            ContactGroup("a", np.zeros((2, 2)), sigma_m=value)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True])
def test_known_distance_sigma_m_requires_positive_finite_metres(value):
    with pytest.raises(ValueError):
        KnownDistance("a", "b", 0.05, sigma_m=value)


def test_sigma_weighting_downweights_the_noisier_group():
    clean, noisy, distance = noisy_groups()
    solver = JointZeroCalibrator(PlanarArm(), residual_tolerance=0.02)
    equal = solver.calibrate_groups(
        [clean, noisy], distances=[distance], distance_tolerance=0.02
    )
    weighted = solver.calibrate_groups(
        [
            ContactGroup(clean.name, clean.samples, sigma_m=0.0005),
            ContactGroup(noisy.name, noisy.samples, sigma_m=0.01),
        ],
        distances=[distance],
        distance_tolerance=0.02,
    )
    true = -0.025
    assert equal.converged and weighted.converged
    assert abs(weighted.offsets_rad[1] - true) < abs(equal.offsets_rad[1] - true) / 50
    assert weighted.evaluation.groups[0].sigma_m == 0.0005
    assert weighted.evaluation.groups[1].sigma_m == 0.01


def test_sigma_weighted_state_rows_match_finite_differences():
    clean, noisy, distance = noisy_groups()
    groups = [
        ContactGroup(clean.name, clean.samples, sigma_m=0.001),
        ContactGroup(noisy.name, noisy.samples, sigma_m=0.02),
    ]
    distance = KnownDistance(
        distance.first, distance.second, distance.distance_m, sigma_m=0.003
    )
    solver = JointZeroCalibrator(PlanarArm())
    q, plan = solver._prepare_groups(groups, [distance], "rad")
    active = np.array([0, 1])
    state = solver._group_state(q, active, plan)
    for column in active:
        delta = np.zeros(2)
        delta[column] = 1e-6
        plus = solver._group_state(q + delta, active, plan)
        minus = solver._group_state(q - delta, active, plan)
        derivative = -(plus.error - minus.error) / 2e-6
        column_values = state.matrix[:, column]
        # Distance rows below the roundoff floor are deliberately zeroed
        # (a common rigid rotation cannot change centre distance); the finite
        # difference must agree on every row that carries information.
        significant = column_values != 0
        np.testing.assert_allclose(
            column_values[significant], derivative[significant], atol=1e-9
        )
        np.testing.assert_array_less(
            np.abs(derivative[~significant]),
            max(1e-9, 1e-8 * np.max(np.abs(column_values))),
        )


def test_distance_sigma_downweights_a_wrong_distance_pull():
    arm = PlanarArm()
    true = np.array([[0.0, 0.4], [1.1, -0.5]])
    points = [arm.position_and_jacobian(q)[0].copy() for q in true]
    distance = float(np.linalg.norm(points[0] - points[1]))
    # Repeated samples give the groups no within-group rank; only the distance
    # term constrains the elbow offset.
    groups = [
        ContactGroup(str(i), np.tile(q - [0, 0.06], (3, 1))) for i, q in enumerate(true)
    ]
    solver = JointZeroCalibrator(arm)
    equal = solver.calibrate_groups(
        groups, distances=[KnownDistance("0", "1", distance + 0.05)]
    )
    weighted = solver.calibrate_groups(
        groups, distances=[KnownDistance("0", "1", distance + 0.05, sigma_m=1e6)]
    )
    assert not equal.converged and not weighted.converged
    assert equal.rank == 1 and weighted.rank == 1
    # A huge sigma nearly disables the distance term, so the fit barely moves
    # from the initial zero offsets while the equal-weight fit chases it.
    assert abs(weighted.offsets_rad[1]) < abs(equal.offsets_rad[1]) / 5
    assert abs(weighted.offsets_rad[1] - 0.06) >= abs(equal.offsets_rad[1] - 0.06)


def test_sigma_reports_round_trip_as_strict_json():
    clean, noisy, distance = noisy_groups()
    groups = [
        ContactGroup(clean.name, clean.samples, sigma_m=0.001),
        ContactGroup(noisy.name, noisy.samples, sigma_m=0.02),
    ]
    distance = KnownDistance(
        distance.first, distance.second, distance.distance_m, sigma_m=0.003
    )
    solver = JointZeroCalibrator(PlanarArm(), residual_tolerance=0.02)
    result = solver.calibrate_groups(
        groups, distances=[distance], distance_tolerance=0.02
    )
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert [g["sigma_m"] for g in payload["evaluation"]["groups"]] == [0.001, 0.02]
    assert payload["evaluation"]["distances"][0]["sigma_m"] == 0.003
    report = solver.evaluate_groups(groups, offsets_rad=[0, 0], distances=[distance])
    assert [g["sigma_m"] for g in report.to_dict()["groups"]] == [0.001, 0.02]


def test_candidate_ranking_scales_rows_by_group_sigma():
    clean, noisy, _ = noisy_groups()
    solver = JointZeroCalibrator(PlanarArm())
    candidates = np.array([[0.2, 1.6], [0.4, 1.5], [-0.3, 1.7]])
    plain = solver.rank_contact_candidates(
        [clean, noisy], candidates, group_name="clean"
    )
    weighted = solver.rank_contact_candidates(
        [
            ContactGroup(clean.name, clean.samples, sigma_m=0.001),
            ContactGroup(noisy.name, noisy.samples, sigma_m=0.02),
        ],
        candidates,
        group_name="clean",
    )
    # A smaller sigma scales the group rows up, so the reported spectrum grows.
    assert weighted.baseline.smallest_observable_singular_value > (
        plain.baseline.smallest_observable_singular_value
    )
    assert len(weighted.candidates) == len(candidates)
    assert weighted.baseline.rank == plain.baseline.rank


@pytest.mark.parametrize("sigma", [0.01, 100.0])
def test_single_group_sigma_matches_equivalent_damping(sigma):
    clean, _, _ = noisy_groups()
    damping = 0.5
    weighted = JointZeroCalibrator(
        PlanarArm(), damping=damping, max_iterations=1
    ).calibrate_groups([ContactGroup("clean", clean.samples, sigma_m=sigma)])
    # Scaling J and r by 1/sigma is equivalent to scaling lambda by sigma².
    reference = JointZeroCalibrator(
        PlanarArm(), damping=damping * sigma**2, max_iterations=1
    ).calibrate(clean.samples)
    np.testing.assert_allclose(weighted.offsets_rad, reference.offsets_rad, atol=1e-14)
    np.testing.assert_allclose(
        weighted.singular_values * sigma, reference.singular_values, rtol=1e-12
    )
    assert weighted.corrected_point_rms == pytest.approx(reference.corrected_point_rms)


def test_single_group_sigma_cannot_hide_physical_residual():
    clean, _, _ = noisy_groups()
    fit = JointZeroCalibrator(PlanarArm(), max_iterations=1).calibrate_groups(
        [ContactGroup("clean", clean.samples, sigma_m=1e12)]
    )
    assert not fit.converged
    assert fit.corrected_point_rms > 1e-3
    assert fit.message == "contact group residual did not reach tolerance"
