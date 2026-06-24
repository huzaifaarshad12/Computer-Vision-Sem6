"""
severity.py — Module 5
Turns the fraction of the image covered by fire/smoke masks into a
human-readable severity rating. Thresholds are tunable.
"""

# (max_fraction_inclusive, level, color_hex)
LEVELS = [
    (0.00, "NONE", "#10b981"),
    (0.05, "LOW", "#22c55e"),
    (0.20, "MEDIUM", "#f59e0b"),
    (0.45, "HIGH", "#f97316"),
    (1.01, "CRITICAL", "#ef4444"),
]


def score(fire_fraction):
    """fire_fraction in [0,1] -> dict(level, color, fraction, percent)."""
    frac = max(0.0, min(1.0, float(fire_fraction)))
    for upper, level, color in LEVELS:
        if frac <= upper:
            return {"level": level, "color": color,
                    "fraction": frac, "percent": round(frac * 100, 1)}
    return {"level": "CRITICAL", "color": "#ef4444",
            "fraction": frac, "percent": round(frac * 100, 1)}
