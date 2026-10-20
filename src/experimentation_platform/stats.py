"""Statistics in pure stdlib — no scipy, no numpy.

An experimentation platform's numbers must be reproducible anywhere,
including a locked-down analysis environment, so every distribution
function here is built from ``math`` primitives:

- Normal CDF from ``math.erf``; its inverse by bisection.
- Student's t CDF from the regularized incomplete beta function
  (Lentz's continued fraction, the classic Numerical Recipes scheme).
- Chi-square survival function from the regularized upper incomplete
  gamma (series / continued-fraction split).

Implemented tests: the pooled two-proportion z-test, Welch's t-test,
and the chi-square goodness-of-fit test used for sample-ratio
mismatch (SRM). Plus closed-form sample-size calculations for
planning. All formulas are the textbook ones and are pinned in the
test suite against hand-computed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF by bisection on :func:`norm_cdf`."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Incomplete beta -> Student's t
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    ib = regularized_beta(x, df / 2.0, 0.5)
    if t > 0:
        return 1.0 - 0.5 * ib
    if t < 0:
        return 0.5 * ib
    return 0.5


def t_ppf(p: float, df: float) -> float:
    """Inverse t CDF by bisection on :func:`t_cdf`."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Incomplete gamma -> chi-square
# ---------------------------------------------------------------------------


def _gamma_p_series(a: float, x: float) -> float:
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b if abs(b) > fpmin else 1.0 / fpmin
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(statistic: float, df: float) -> float:
    """Survival function P(X > statistic) for chi-square with ``df`` dof."""
    if statistic <= 0:
        return 1.0
    a, x = df / 2.0, statistic / 2.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_cf(a, x)


# ---------------------------------------------------------------------------
# Tests of hypotheses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProportionResult:
    control_rate: float
    treatment_rate: float
    absolute_effect: float
    relative_effect: float | None
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def two_proportion_ztest(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ProportionResult:
    """Pooled two-proportion z-test (two-sided).

    The test statistic uses the pooled standard error (correct under
    H0); the confidence interval for the difference uses the unpooled
    standard error (correct under H1) — the standard pairing.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("sample sizes must be positive")
    p_c = control_successes / control_n
    p_t = treatment_successes / treatment_n
    pooled = (control_successes + treatment_successes) / (control_n + treatment_n)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    diff = p_t - p_c
    if se_pooled == 0.0:
        z, p_value = 0.0, 1.0
    else:
        z = diff / se_pooled
        p_value = 2.0 * norm_sf(abs(z))
    se_unpooled = math.sqrt(
        p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n
    )
    crit = norm_ppf(1 - alpha / 2)
    return ProportionResult(
        control_rate=p_c,
        treatment_rate=p_t,
        absolute_effect=diff,
        relative_effect=(diff / p_c) if p_c > 0 else None,
        z=z,
        p_value=p_value,
        ci_low=diff - crit * se_unpooled,
        ci_high=diff + crit * se_unpooled,
        alpha=alpha,
    )


@dataclass(frozen=True)
class WelchResult:
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float | None
    t: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def welch_ttest_from_stats(
    control_mean: float,
    control_var: float,
    control_n: int,
    treatment_mean: float,
    treatment_var: float,
    treatment_n: int,
    alpha: float = 0.05,
) -> WelchResult:
    """Welch's t-test from summary statistics (unequal variances)."""
    if control_n < 2 or treatment_n < 2:
        raise ValueError("Welch's t-test needs at least 2 observations per group")
    diff = treatment_mean - control_mean
    v_c, v_t = control_var / control_n, treatment_var / treatment_n
    se = math.sqrt(v_c + v_t)
    if se == 0.0:
        t_stat, df, p_value = 0.0, float(control_n + treatment_n - 2), 1.0
        ci_low = ci_high = diff
        return WelchResult(
            control_mean, treatment_mean, diff,
            (diff / control_mean) if control_mean != 0 else None,
            t_stat, df, p_value, ci_low, ci_high, alpha,
        )
    t_stat = diff / se
    # Welch–Satterthwaite degrees of freedom.
    df = (v_c + v_t) ** 2 / (
        v_c**2 / (control_n - 1) + v_t**2 / (treatment_n - 1)
    )
    p_value = 2.0 * (1.0 - t_cdf(abs(t_stat), df))
    crit = t_ppf(1 - alpha / 2, df)
    return WelchResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        absolute_effect=diff,
        relative_effect=(diff / control_mean) if control_mean != 0 else None,
        t=t_stat,
        df=df,
        p_value=p_value,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        alpha=alpha,
    )


def welch_ttest(
    control: list[float], treatment: list[float], alpha: float = 0.05
) -> WelchResult:
    """Welch's t-test from raw samples."""

    def mean_var(xs: list[float]) -> tuple[float, float]:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return mean, var

    cm, cv = mean_var(control)
    tm, tv = mean_var(treatment)
    return welch_ttest_from_stats(cm, cv, len(control), tm, tv, len(treatment), alpha)


@dataclass(frozen=True)
class SRMResult:
    chi2: float
    df: int
    p_value: float
    threshold: float
    observed: tuple[int, ...]
    expected: tuple[float, ...]

    @property
    def passed(self) -> bool:
        """SRM uses a strict threshold: p below it means the split
        itself is broken, however the metrics look."""
        return self.p_value >= self.threshold


def srm_check(
    observed: list[int], weights: list[float], threshold: float = 0.001
) -> SRMResult:
    """Chi-square goodness-of-fit of exposure counts vs planned weights."""
    if len(observed) != len(weights) or len(observed) < 2:
        raise ValueError("observed and weights must align and have >= 2 cells")
    total = sum(observed)
    if total <= 0:
        raise ValueError("no exposures to check")
    expected = [total * w for w in weights]
    chi2 = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    df = len(observed) - 1
    return SRMResult(
        chi2=chi2,
        df=df,
        p_value=chi2_sf(chi2, df),
        threshold=threshold,
        observed=tuple(observed),
        expected=tuple(expected),
    )


# ---------------------------------------------------------------------------
# Planning: closed-form sample sizes
# ---------------------------------------------------------------------------


def required_n_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-variant n to detect an absolute lift in a proportion.

    n = (z_{1-a/2} + z_power)^2 * (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2
    """
    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive")
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("rates must lie in (0, 1)")
    z = norm_ppf(1 - alpha / 2) + norm_ppf(power)
    n = z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / minimum_detectable_effect**2
    return math.ceil(n)


def required_n_continuous(
    minimum_detectable_effect: float,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-variant n to detect a mean shift: 2 (z_a + z_b)^2 s^2 / d^2."""
    if minimum_detectable_effect <= 0 or std_dev <= 0:
        raise ValueError("effect and std_dev must be positive")
    z = norm_ppf(1 - alpha / 2) + norm_ppf(power)
    return math.ceil(2 * z**2 * std_dev**2 / minimum_detectable_effect**2)


def achieved_power_proportions(
    baseline_rate: float, effect: float, n_per_variant: int, alpha: float = 0.05
) -> float:
    """Approximate power at a given per-variant n (normal approximation)."""
    if n_per_variant <= 0 or effect == 0:
        return 0.0
    p1, p2 = baseline_rate, baseline_rate + effect
    se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_variant)
    return norm_cdf(abs(effect) / se - norm_ppf(1 - alpha / 2))
