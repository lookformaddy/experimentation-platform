import math

import pytest

from experimentation_platform.stats import (
    chi2_sf,
    norm_cdf,
    norm_ppf,
    required_n_continuous,
    required_n_proportions,
    srm_check,
    t_cdf,
    two_proportion_ztest,
    welch_ttest,
)


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975002, abs=1e-5)
    assert norm_cdf(-1.0) == pytest.approx(0.158655, abs=1e-5)


def test_norm_ppf_inverts_cdf():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    for p in (0.01, 0.1, 0.9, 0.99):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_two_proportion_ztest_hand_computed():
    # control 100/1000 vs treatment 130/1000 (verified by hand / scipy)
    r = two_proportion_ztest(100, 1000, 130, 1000)
    assert r.control_rate == pytest.approx(0.10)
    assert r.treatment_rate == pytest.approx(0.13)
    assert r.absolute_effect == pytest.approx(0.03)
    assert r.relative_effect == pytest.approx(0.30)
    assert r.z == pytest.approx(2.102741, abs=1e-5)
    assert r.p_value == pytest.approx(0.035488, abs=1e-5)
    assert r.ci_low == pytest.approx(0.002068, abs=1e-5)
    assert r.ci_high == pytest.approx(0.057932, abs=1e-5)
    assert r.significant


def test_two_proportion_ztest_no_effect():
    r = two_proportion_ztest(100, 1000, 100, 1000)
    assert r.z == 0.0
    assert r.p_value == 1.0
    assert not r.significant


def test_two_proportion_ztest_rejects_empty():
    with pytest.raises(ValueError):
        two_proportion_ztest(0, 0, 1, 10)


def test_t_cdf_known_values():
    assert t_cdf(0.0, 10) == pytest.approx(0.5)
    # t = -3.6742346 with df=4 -> CDF 0.010656 (scipy reference)
    assert t_cdf(-3.6742346141747673, 4) == pytest.approx(0.010656, abs=1e-5)


def test_welch_ttest_hand_computed():
    # control [1,2,3] vs treatment [4,5,6]: the platform reports
    # treatment - control, so t = +3.6742, df = 4, p = 0.021312 (scipy,
    # sign flipped from ttest_ind's first-minus-second convention)
    r = welch_ttest([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert r.t == pytest.approx(3.674235, abs=1e-5)
    assert r.df == pytest.approx(4.0)
    assert r.p_value == pytest.approx(0.021312, abs=1e-5)
    assert r.absolute_effect == pytest.approx(3.0)
    assert r.significant


def test_welch_ttest_identical_samples():
    r = welch_ttest([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r.t == 0.0
    assert r.p_value == 1.0


def test_chi2_sf_known_values():
    assert chi2_sf(0.0, 1) == 1.0
    # chi2 = 3.8415 with df=1 is the 95th percentile
    assert chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-4)


def test_srm_balanced_split_passes():
    r = srm_check([500, 500], [0.5, 0.5])
    assert r.chi2 == 0.0
    assert r.p_value == 1.0
    assert r.passed


def test_srm_skewed_split_fails():
    # [600, 400] vs 50/50: chi2 = 40 exactly, p ~ 2.5e-10 (scipy)
    r = srm_check([600, 400], [0.5, 0.5])
    assert r.chi2 == pytest.approx(40.0)
    assert r.p_value == pytest.approx(2.54e-10, rel=0.05)
    assert not r.passed


def test_srm_respects_planned_weights():
    r = srm_check([700, 300], [0.7, 0.3])
    assert r.chi2 == pytest.approx(0.0)
    assert r.passed


def test_required_n_proportions_hand_computed():
    # (z.975 + z.8)^2 * (0.09 + 0.1056) / 0.02^2 = 3838.10 -> ceil 3839
    assert required_n_proportions(0.10, 0.02) == 3839
    # baseline 0.20, MDE 0.05 -> 1091
    assert required_n_proportions(0.20, 0.05) == 1091


def test_required_n_proportions_monotonic_in_mde():
    assert required_n_proportions(0.10, 0.01) > required_n_proportions(0.10, 0.02)


def test_required_n_continuous_hand_computed():
    # 2 * (z.975 + z.8)^2 * 10^2 / 2^2 = 392.44 -> ceil 393
    assert required_n_continuous(2.0, 10.0) == 393


def test_sample_size_rejects_bad_inputs():
    with pytest.raises(ValueError):
        required_n_proportions(0.10, 0.0)
    with pytest.raises(ValueError):
        required_n_continuous(-1.0, 10.0)
