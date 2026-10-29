"""Frozen width mapping for scaling validation; independent of observed losses."""

from fractions import Fraction

FAMILIES = ("b0", "b1", "k3", "rrp2_learn")
SCALES = ("10m", "100m")
ARTIFACTS = {
    "10m": "38817d20b46a68322b813a02299301724c7f030fb950b0d6d1540600b5e7606c",
    "100m": "67962070e8a2e9b633bcf7d1cf0509ec36f32f2b95858b6586845f4d5b9099cf",
}
# The grid and 100,007,936-label calibration follow the original scale recipes.
LRS = ("0.0003", "0.001", "0.003")


def mapped_widths(hb, family):
    if family not in ("k3", "rrp2_learn") or hb < 1:
        raise ValueError("Expected a positive budget and a mapped family")
    unit, coefficient, alpha = (
        (4, Fraction(11, 4), Fraction(232, 853))
        if family == "k3"
        else (16, Fraction(23, 8), Fraction(192, 853))
    )
    candidates = []
    # Every positive-anchor feasible width, including nearest-boundary rounding.
    for hd in range(unit, int(3 * hb / coefficient) + unit, unit):
        target = (3 * hb - coefficient * hd) / 3
        lower = target.numerator // target.denominator
        for ha in {lower, lower + 1}:
            if ha > 0:
                candidates.append(
                    (
                        abs(3 * ha + coefficient * hd - 3 * hb),
                        abs(Fraction(hd, hb) - alpha),
                        hd,
                        ha,
                    )
                )
    if not candidates:
        raise ValueError("No positive-anchor mapping fits this budget")
    error, ratio_error, hd, ha = min(candidates)
    return {
        "ha": ha,
        "hd": hd,
        "projection_budget_error": float(error),
        "ratio_error": float(ratio_error),
        "target_hd_over_hb": float(alpha),
    }


def scaling_architecture(scale, family):
    from .models import architecture

    if scale not in SCALES or family not in FAMILIES:
        raise ValueError("Unregistered scaling experiment")
    a = architecture(scale)
    h0 = a.ff_dim
    hb = (5 * h0 + 3) // 6
    a.initialization_reference_width = h0
    if family == "b1":
        a.ff_dim = hb
    elif family in ("k3", "rrp2_learn"):
        mapping = mapped_widths(hb, family)
        a.anchor_width = mapping["ha"]
        a.ff_dim = mapping["ha"] + mapping["hd"]
        a.ffn_kind = "s3_k3" if family == "k3" else "rrp2_learn"
        a.local_group_size = 4 if family == "k3" else 16
        a.local_outputs = 3 if family == "k3" else 2
        a.local_mode = "shared" if family == "k3" else "learnable"
        # Legacy static branch divisibility is unrelated to local projection.
        a.branches = 1
    return a


def scaling_recipe(scale, family):
    from .config_registry import formal_recipe

    config = formal_recipe(scale)
    config.architecture = scaling_architecture(scale, family)
    config.paired_initialization = True
    config.debug.deterministic = True
    config.audit_batches = True
    config.static_diagnostics_freq = 100
    # Same microbatch and accumulation across all four families at each scale.
    config.training.local_batch_size = 16 if scale == "10m" else 4
    return config
