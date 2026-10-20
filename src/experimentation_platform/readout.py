"""Decision readouts and the peeking guard.

A readout combines four independent checks into one decision:

1. **SRM.** If the exposure split does not match the planned weights
   (chi-square p < 0.001), the randomization or logging is broken and
   *no* metric result can be trusted -> INCONCLUSIVE (invalid).
2. **Guardrails.** A statistically significant guardrail degradation
   beyond its tolerance vetoes shipping -> NO-SHIP, even when the
   primary metric wins. Winning the metric and breaking the product
   is not a win.
3. **Primary metric.** Significant and positive (in the metric's
   good direction) -> SHIP; significant and negative -> NO-SHIP.
4. **Everything else** — underpowered, null, or peeking-flagged —
   is INCONCLUSIVE, with the reason stated.

Peeking guard — stated honestly: this platform does **not** implement
a full sequential test (no mixture sequential probability ratio test,
no alpha-spending *inference*). What it implements is a pragmatic
guard in the O'Brien–Fleming spirit: at information fraction
``t = n_now / n_planned``, the OBF critical value is
``z_{1-a/2} / sqrt(t)``. A result that is naively significant
(p < alpha) but does not clear the OBF boundary at the current
information fraction is flagged ``peeking_risk`` and cannot produce a
SHIP decision — the honest recommendation is "keep running". At
t = 1 the boundary equals the usual critical value, so a planned-end
read is unaffected. That is a guardrail against the classic
"check daily, stop when significant" failure mode, not a replacement
for a properly designed sequential analysis.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from .metrics import MetricResult
from .stats import SRMResult, norm_ppf


class Decision(enum.Enum):
    SHIP = "ship"
    NO_SHIP = "no_ship"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PeekingAssessment:
    information_fraction: float
    looks: int
    naive_significant: bool
    obf_critical_z: float
    observed_z: float | None
    peeking_risk: bool
    warning: str | None = None


def assess_peeking(
    z_value: float | None,
    p_value: float,
    n_current: int,
    n_required: int,
    looks: int = 1,
    alpha: float = 0.05,
) -> PeekingAssessment:
    """OBF-style early-read guard. See module docstring for scope."""
    if n_required and n_required > 0:
        t = min(1.0, n_current / n_required)
    else:
        t = 1.0
    t = max(t, 1e-6)
    obf_crit = norm_ppf(1 - alpha / 2) / math.sqrt(t)
    naive_sig = p_value < alpha
    risk = bool(
        naive_sig
        and z_value is not None
        and abs(z_value) < obf_crit
        and (t < 1.0 or looks > 1)
    )
    warning = None
    if risk:
        warning = (
            f"Result is naively significant (p={p_value:.4f}) at information "
            f"fraction {t:.2f} after {looks} look(s), but does not clear the "
            f"O'Brien-Fleming-style boundary |z| >= {obf_crit:.2f}. "
            "Treat as peeking risk: keep running to the planned sample size."
        )
    elif t < 0.2 and looks > 1:
        warning = (
            f"Very early read (information fraction {t:.2f}) after {looks} "
            "looks: estimates are unstable even when not flagged."
        )
    return PeekingAssessment(
        information_fraction=t,
        looks=looks,
        naive_significant=naive_sig,
        obf_critical_z=obf_crit,
        observed_z=z_value,
        peeking_risk=risk,
        warning=warning,
    )


@dataclass
class GuardrailBreach:
    metric: str
    treatment: str
    absolute_effect: float
    p_value: float
    tolerance: float


@dataclass
class Readout:
    experiment_id: str
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    srm: SRMResult | None = None
    guardrail_breaches: list[GuardrailBreach] = field(default_factory=list)
    peeking: PeekingAssessment | None = None
    n_current: int = 0
    n_required: int | None = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "decision": self.decision.value,
            "reasons": self.reasons,
            "srm": (
                {
                    "chi2": self.srm.chi2,
                    "df": self.srm.df,
                    "p_value": self.srm.p_value,
                    "passed": self.srm.passed,
                    "observed": list(self.srm.observed),
                    "expected": list(self.srm.expected),
                }
                if self.srm
                else None
            ),
            "guardrail_breaches": [
                {
                    "metric": b.metric,
                    "treatment": b.treatment,
                    "absolute_effect": b.absolute_effect,
                    "p_value": b.p_value,
                    "tolerance": b.tolerance,
                }
                for b in self.guardrail_breaches
            ],
            "peeking": (
                {
                    "information_fraction": self.peeking.information_fraction,
                    "looks": self.peeking.looks,
                    "peeking_risk": self.peeking.peeking_risk,
                    "obf_critical_z": self.peeking.obf_critical_z,
                    "warning": self.peeking.warning,
                }
                if self.peeking
                else None
            ),
            "n_current": self.n_current,
            "n_required": self.n_required,
        }


def build_readout(
    experiment_id: str,
    primary: MetricResult,
    guardrails: dict[str, MetricResult],
    srm: SRMResult | None,
    n_required: int | None = None,
    looks: int = 1,
    alpha: float = 0.05,
) -> Readout:
    reasons: list[str] = []
    n_current = primary.total_exposed

    # -- Guardrail breaches (computed first; they veto) -------------------
    breaches: list[GuardrailBreach] = []
    for name, result in guardrails.items():
        spec = result.metric
        for comp in result.comparisons:
            effect = comp.absolute_effect
            degraded = (
                effect < -spec.guardrail_tolerance
                if spec.higher_is_better
                else effect > spec.guardrail_tolerance
            )
            if degraded and comp.significant:
                breaches.append(
                    GuardrailBreach(
                        metric=name,
                        treatment=comp.treatment,
                        absolute_effect=effect,
                        p_value=comp.p_value,
                        tolerance=spec.guardrail_tolerance,
                    )
                )

    # -- Peeking assessment on the primary comparison ----------------------
    peeking: PeekingAssessment | None = None
    primary_comp = primary.comparisons[0] if primary.comparisons else None
    if primary_comp is not None:
        z_value = getattr(primary_comp.result, "z", None)
        if z_value is None and hasattr(primary_comp.result, "t"):
            z_value = primary_comp.result.t  # t approximates z for the guard
        per_variant_required = (n_required or 0) * max(
            1, len(primary.per_variant)
        )
        peeking = assess_peeking(
            z_value=z_value,
            p_value=primary_comp.p_value,
            n_current=n_current,
            n_required=per_variant_required if n_required else 0,
            looks=looks,
            alpha=alpha,
        )

    # -- Decision ------------------------------------------------------------
    if srm is not None and not srm.passed:
        reasons.append(
            f"SRM check failed (chi2={srm.chi2:.2f}, p={srm.p_value:.2g} < "
            f"{srm.threshold}): the exposure split {list(srm.observed)} does "
            "not match the planned weights. Randomization or logging is "
            "broken; metric results are not trustworthy."
        )
        return Readout(
            experiment_id, Decision.INCONCLUSIVE, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    if breaches:
        for b in breaches:
            reasons.append(
                f"Guardrail {b.metric!r} breached for {b.treatment}: effect "
                f"{b.absolute_effect:+.4f} beyond tolerance "
                f"{b.tolerance:.4f} (p={b.p_value:.4f})."
            )
        reasons.append("Guardrail breach vetoes shipping.")
        return Readout(
            experiment_id, Decision.NO_SHIP, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    if primary_comp is None:
        reasons.append("No primary-metric comparison could be computed "
                       "(missing or insufficient data).")
        return Readout(
            experiment_id, Decision.INCONCLUSIVE, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    spec = primary.metric
    effect = primary_comp.absolute_effect
    good_direction = effect > 0 if spec.higher_is_better else effect < 0

    if primary_comp.significant and peeking is not None and peeking.peeking_risk:
        reasons.append(peeking.warning or "Peeking risk flagged.")
        return Readout(
            experiment_id, Decision.INCONCLUSIVE, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    if primary_comp.significant and good_direction:
        reasons.append(
            f"Primary metric {spec.name!r} improved significantly for "
            f"{primary_comp.treatment}: effect {effect:+.4f} "
            f"(p={primary_comp.p_value:.4f}), no guardrail breaches, SRM ok."
        )
        return Readout(
            experiment_id, Decision.SHIP, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    if primary_comp.significant and not good_direction:
        reasons.append(
            f"Primary metric {spec.name!r} significantly worsened for "
            f"{primary_comp.treatment}: effect {effect:+.4f} "
            f"(p={primary_comp.p_value:.4f})."
        )
        return Readout(
            experiment_id, Decision.NO_SHIP, reasons, srm, breaches,
            peeking, n_current, n_required,
        )

    if n_required is not None:
        per_variant_n = (
            n_current // max(1, len(primary.per_variant))
            if primary.per_variant
            else 0
        )
        reasons.append(
            f"Primary metric not significant (p={primary_comp.p_value:.4f}); "
            f"~{per_variant_n} exposed per variant vs ~{n_required} required "
            "per variant for the planned minimum detectable effect."
        )
    else:
        reasons.append(
            f"Primary metric not significant (p={primary_comp.p_value:.4f})."
        )
    return Readout(
        experiment_id, Decision.INCONCLUSIVE, reasons, srm, breaches,
        peeking, n_current, n_required,
    )
