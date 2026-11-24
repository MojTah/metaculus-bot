"""Free backtest: compare the bot with the Metaculus community prediction (CP).

Questions are OPEN binary questions with 30+ forecasters, so no outcome is
known yet and the model cannot have learned it. The score is the expected
baseline score (forecasting-tools BinaryReport.expected_baseline_score): the
baseline score the forecast would get if the CP were the true probability.
50% scores 0; copying the CP scores the maximum. Higher is better.

Research is run once per question and reused by every model, so the models
are compared on identical inputs. Nothing is published.

    uv run python backtest.py --n 20 --models gemini/gemini-flash-lite-latest,gemini/gemma-4-31b-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone

import dotenv
from forecasting_tools import BinaryQuestion, BinaryReport, MetaculusClient

from bot_config import ThrottledLlm, build_llms
from main import FallTemplateBot2026

dotenv.load_dotenv()
logger = logging.getLogger("backtest")

# Gemma on the free tier has 16K input tokens per minute: one forecast prompt
# is about 3-5K tokens, so space Gemma calls by 20 s.
_MIN_INTERVAL_OVERRIDES = {"gemma": 20.0}


def expected_baseline_score(p: float, c: float) -> float:
    return 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))


def scale_logit(p: float, a: float) -> float:
    p = min(max(p, 0.01), 0.99)
    x = a * math.log(p / (1 - p))
    return min(max(1 / (1 + math.exp(-x)), 0.01), 0.99)


def best_scale(pairs: list[tuple[float, float]]) -> float:
    grid = [0.5 + 0.05 * i for i in range(41)]  # 0.5 .. 2.5
    return max(
        grid,
        key=lambda a: sum(expected_baseline_score(scale_logit(p, a), c) for p, c in pairs),
    )


def loo_scaled_score(pairs: list[tuple[float, float]]) -> float | None:
    """Leave-one-out score of a fitted logit scale (honest estimate of the gain)."""
    if len(pairs) < 5:
        return None
    scores = []
    for i, (p, c) in enumerate(pairs):
        a = best_scale(pairs[:i] + pairs[i + 1 :])
        scores.append(expected_baseline_score(scale_logit(p, a), c))
    return statistics.mean(scores)


def _diagnose_missing_cp() -> None:
    """Log what the API returns when no question has a community prediction."""
    from forecasting_tools import ApiFilter

    qs = asyncio.run(
        MetaculusClient().get_questions_matching_filter(
            ApiFilter(allowed_statuses=["open"], allowed_types=["binary"], num_forecasters_gte=30),
            num_questions=5,
            error_if_question_target_missed=False,
        )
    )
    logger.warning("Diagnose: %d questions without the CP filter", len(qs))
    for q in qs[:3]:
        agg = (q.api_json.get("question") or {}).get("aggregations")
        if isinstance(agg, dict):
            shape = {k: (sorted(v.keys()) if isinstance(v, dict) else type(v).__name__) for k, v in agg.items()}
            latest = {k: (v.get("latest") if isinstance(v, dict) else None) for k, v in agg.items()}
        else:
            shape, latest = type(agg).__name__, None
        logger.warning("Diagnose %s: aggregations=%s latest=%s cp_hidden=%s",
                       q.page_url, shape, str(latest)[:300],
                       (q.api_json.get("question") or {}).get("cp_reveal_time"))


class CachedResearchBot(FallTemplateBot2026):
    research_cache: dict[str, str] = {}

    async def run_research(self, question):  # type: ignore[override]
        key = question.page_url or question.question_text
        if key not in self.research_cache:
            self.research_cache[key] = await super().run_research(question)
        return self.research_cache[key]


def variant_llms(model: str, base: dict) -> dict:
    interval = next(
        (v for k, v in _MIN_INTERVAL_OVERRIDES.items() if k in model.lower()), None
    )
    # No backup models here: every answer must come from the model under test.
    default = ThrottledLlm(
        model=model, temperature=0.3, timeout=180, allowed_tries=4, min_interval=interval
    )
    return {
        "default": default,
        "summarizer": base["summarizer"],
        "parser": base["parser"],
        "researcher": base["researcher"],
    }


async def run(n: int, models: list[str], predictions: int, out_dir: str) -> dict:
    base = build_llms()
    if base is None:
        raise SystemExit("No LLM key set")
    questions: list[BinaryQuestion] = await asyncio.to_thread(
        MetaculusClient().get_benchmark_questions,
        n,
        None,
        365,
        30,
        False,
    )
    questions = [q for q in questions if q.community_prediction_at_access_time is not None]
    if not questions:
        await asyncio.to_thread(_diagnose_missing_cp)
    logger.info("Backtest on %d open binary questions", len(questions))

    preds: dict[str, dict[str, float]] = {}
    errors: dict[str, int] = {}
    for model in models:
        bot = CachedResearchBot(
            research_reports_per_question=1,
            predictions_per_research_report=predictions,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
            skip_previously_forecasted_questions=False,
            llms=variant_llms(model, base),
        )
        started = datetime.now(timezone.utc)
        reports = await bot.forecast_questions(questions, return_exceptions=True)
        preds[model] = {}
        errors[model] = 0
        for q, r in zip(questions, reports):
            if isinstance(r, BinaryReport):
                preds[model][q.page_url] = float(r.prediction)
            else:
                errors[model] += 1
                logger.warning("%s failed on %s: %s", model, q.page_url, str(r)[:300])
        logger.info(
            "%s done in %.0f s, %d errors",
            model,
            (datetime.now(timezone.utc) - started).total_seconds(),
            errors[model],
        )

    rows = []
    for q in questions:
        c = float(q.community_prediction_at_access_time)  # type: ignore[arg-type]
        row = {"url": q.page_url, "title": q.question_text[:120], "cp": c, "pred": {}}
        for model in models:
            if q.page_url in preds[model]:
                row["pred"][model] = preds[model][q.page_url]
        got = list(row["pred"].values())
        if len(models) > 1 and len(got) == len(models):
            row["pred"]["ensemble_median"] = statistics.median(got)
        rows.append(row)

    variants = list(models) + (["ensemble_median"] if len(models) > 1 else [])
    summary = {}
    for v in variants:
        pairs = [(r["pred"][v], r["cp"]) for r in rows if v in r["pred"]]
        if not pairs:
            continue
        # Compare every variant on the questions all variants answered.
        summary[v] = {
            "answered": len(pairs),
            "score": statistics.mean(expected_baseline_score(p, c) for p, c in pairs),
            "mean_abs_dev": statistics.mean(abs(p - c) for p, c in pairs),
            "best_logit_scale": best_scale(pairs),
            "loo_scaled_score": loo_scaled_score(pairs),
            "errors": errors.get(v, 0),
        }
    common = [r for r in rows if all(v in r["pred"] for v in variants)]
    for v in variants:
        if v in summary and common:
            summary[v]["score_common"] = statistics.mean(
                expected_baseline_score(r["pred"][v], r["cp"]) for r in common
            )
    ceiling = statistics.mean(expected_baseline_score(min(max(r["cp"], 0.01), 0.99), r["cp"]) for r in common) if common else None
    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "n_common": len(common),
        "predictions_per_question": predictions,
        "ceiling_copy_cp": ceiling,
        "baseline_50pct": 0.0,
        "summary": summary,
        "rows": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "backtest.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def render(result: dict) -> str:
    lines = [
        f"BACKTEST {result['date'][:16]} questions={result['n_questions']} "
        f"common={result['n_common']} predictions/question={result['predictions_per_question']}",
        f"ceiling (copy CP) = {result['ceiling_copy_cp']}, 50% baseline = 0",
        "variant | answered | score | score_common | mean|p-cp| | best_scale | LOO scaled",
    ]
    for v, s in result["summary"].items():
        lines.append(
            f"{v} | {s['answered']} | {s['score']:.2f} | {s.get('score_common', float('nan')):.2f} | "
            f"{s['mean_abs_dev']:.3f} | {s['best_logit_scale']:.2f} | {s['loo_scaled_score']}"
        )
    lines.append("ROWS: cp | predictions | title")
    for r in result["rows"]:
        p = " ".join(f"{k.split('/')[-1]}={v:.2f}" for k, v in r["pred"].items())
        lines.append(f"{r['cp']:.2f} | {p} | {r['title'][:70]}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--models",
        default="gemini/gemini-flash-lite-latest,gemini/gemini-3.1-flash-lite,gemini/gemma-4-31b-it",
    )
    ap.add_argument("--predictions", type=int, default=3)
    ap.add_argument("--out", default="reports/backtest")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    res = asyncio.run(run(a.n, models, a.predictions, a.out))
    text = render(res)
    with open(os.path.join(a.out, "backtest.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
