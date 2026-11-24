import math

from backtest import best_scale, expected_baseline_score, loo_scaled_score, render, scale_logit


def test_expected_baseline_score_basics():
    assert abs(expected_baseline_score(0.5, 0.8)) < 1e-9
    # copying the CP is the best forecast
    c = 0.8
    best = expected_baseline_score(c, c)
    assert all(expected_baseline_score(p, c) <= best + 1e-9 for p in (0.6, 0.7, 0.9, 0.95))


def test_scale_logit_and_fit():
    assert abs(scale_logit(0.5, 2.0) - 0.5) < 1e-9
    assert scale_logit(0.7, 2.0) > 0.7
    # predictions too timid vs CP -> fitted scale > 1
    pairs = [(0.6, 0.8), (0.4, 0.2), (0.7, 0.9), (0.3, 0.1), (0.55, 0.65)]
    assert best_scale(pairs) > 1.2
    assert loo_scaled_score(pairs) is not None
    assert loo_scaled_score(pairs[:3]) is None


def test_render_runs():
    res = {
        "date": "2026-11-24T00:00:00",
        "n_questions": 1,
        "n_common": 1,
        "predictions_per_question": 3,
        "ceiling_copy_cp": 10.0,
        "summary": {
            "m": {"answered": 1, "score": 1.0, "score_common": 1.0, "mean_abs_dev": 0.1,
                  "best_logit_scale": 1.0, "loo_scaled_score": None, "errors": 0}
        },
        "rows": [{"url": "u", "title": "t", "cp": 0.5, "pred": {"gemini/m": 0.4}}],
    }
    assert "BACKTEST" in render(res)
    assert not math.isnan(res["summary"]["m"]["score"])
