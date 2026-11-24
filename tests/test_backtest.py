from backtest import baseline_score, best_scale, brier, loo_scaled_score, render, scale_logit


def test_scores():
    assert abs(baseline_score(0.5, True)) < 1e-9
    assert baseline_score(0.9, True) > 0 > baseline_score(0.9, False)
    assert brier(0.5, False) == 0.25
    assert brier(1.0, True) == 0.0


def test_scale_logit_and_fit():
    assert abs(scale_logit(0.5, 2.0) - 0.5) < 1e-9
    assert scale_logit(0.7, 2.0) > 0.7
    # timid but right -> fitted scale > 1
    pairs = [(0.6, True), (0.4, False), (0.7, True), (0.3, False), (0.65, True), (0.35, False)]
    assert best_scale(pairs) > 1.5
    assert loo_scaled_score(pairs) is not None
    assert loo_scaled_score(pairs[:3]) is None


def test_render_runs():
    res = {
        "date": "2026-11-24T00:00:00", "resolved_after": "2026-09-01", "tournaments": [],
        "n_questions": 1, "n_common": 1, "yes_rate": 1.0, "predictions_per_question": 3,
        "summary": {"m": {"answered": 1, "score": 1.0, "brier": 0.1,
                          "best_logit_scale": 1.0, "loo_scaled_score": None, "errors": 0}},
        "rows": [{"url": "u", "title": "t", "outcome": True, "pred": {"gemini/m": 0.4}}],
    }
    assert "BACKTEST" in render(res)
