"""Free backtest on RESOLVED binary questions (nothing is published).

Why this design:
- Bot accounts cannot see the community prediction (the API returns
  "latest": null), so the score is against the real outcome.
- Web search would find the outcome, so research is OFF: the models forecast
  from the question text only. This measures the model and the prompt, not
  the research. Pick questions that resolved after the models' training
  cutoff (default: scheduled to resolve after 2026-09-01).

Scores per question (higher is better unless noted):
- baseline score = 100 * (log2(p_outcome) + 1); 50% scores 0 (Metaculus scale)
- Brier = (p - outcome)^2 (lower is better)

    uv run python backtest.py --n 30 --models gemini/gemini-flash-lite-latest,gemini/gemma-4-31b-it
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
from forecasting_tools import ApiFilter, BinaryQuestion, BinaryReport, MetaculusClient

from bot_config import ThrottledLlm, build_llms
from main import FallTemplateBot2026

dotenv.load_dotenv()
logger = logging.getLogger("backtest")

# Gemma on the free tier has 16K input tokens per minute.
_MIN_INTERVAL_OVERRIDES = {"gemma": 15.0}


def baseline_score(p: float, outcome: bool) -> float:
    q = p if outcome else 1 - p
    return 100.0 * (math.log2(q) + 1.0)


def brier(p: float, outcome: bool) -> float:
    return (p - (1.0 if outcome else 0.0)) ** 2


def scale_logit(p: float, a: float) -> float:
    p = min(max(p, 0.01), 0.99)
    x = a * math.log(p / (1 - p))
    return min(max(1 / (1 + math.exp(-x)), 0.01), 0.99)


def best_scale(pairs: list[tuple[float, bool]]) -> float:
    grid = [0.5 + 0.05 * i for i in range(41)]  # 0.5 .. 2.5
    return max(grid, key=lambda a: sum(baseline_score(scale_logit(p, a), o) for p, o in pairs))


def loo_scaled_score(pairs: list[tuple[float, bool]]) -> float | None:
    """Leave-one-out score with a fitted logit scale (honest estimate of the gain)."""
    if len(pairs) < 5:
        return None
    scores = []
    for i, (p, o) in enumerate(pairs):
        a = best_scale(pairs[:i] + pairs[i + 1 :])
        scores.append(baseline_score(scale_logit(p, a), o))
    return statistics.mean(scores)


def variant_llms(model: str, base: dict) -> dict:
    interval = next(
        (v for k, v in _MIN_INTERVAL_OVERRIDES.items() if k in model.lower()), None
    )
    # No backup models: every answer must come from the model under test.
    default = ThrottledLlm(
        model=model, temperature=0.3, timeout=180, allowed_tries=4, min_interval=interval
    )
    return {
        "default": default,
        "summarizer": base["summarizer"],
        "parser": base["parser"],
        "researcher": "no_research",
    }


class AnonymousClient(MetaculusClient):
    """Read public question data without the bot token.

    The API hides resolutions and the community prediction from bot accounts.
    """

    def _get_auth_headers(self) -> dict[str, dict[str, str]]:  # type: ignore[override]
        return {"headers": {"Accept-Language": "en"}}


def fetch_questions(n: int, resolved_after: str, tournaments: list[str],
                    anonymous: bool = True) -> list[BinaryQuestion]:
    after = datetime.fromisoformat(resolved_after).replace(tzinfo=timezone.utc)
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        scheduled_resolve_time_gt=after,
        scheduled_resolve_time_lt=datetime.now(timezone.utc),
        allowed_tournaments=tournaments or None,
        group_question_mode="exclude",
    )
    client = AnonymousClient() if anonymous else MetaculusClient()
    qs = asyncio.run(
        client.get_questions_matching_filter(
            api_filter,
            num_questions=n,
            randomly_sample=True,
            error_if_question_target_missed=False,
        )
    )
    kept = [q for q in qs if isinstance(q, BinaryQuestion) and isinstance(q.typed_resolution, bool)]
    if len(kept) < len(qs):
        for q in [q for q in qs if q not in kept][:3]:
            qj = q.api_json.get("question") or {}
            logger.warning(
                "Dropped %s: resolution_string=%r status=%r api resolution=%r keys=%s",
                q.page_url, q.resolution_string, qj.get("status"), qj.get("resolution"),
                sorted(k for k in qj.keys() if "resol" in k or "status" in k),
            )
    return kept


def fetch_manifold_questions(n: int, resolved_after: str, min_bettors: int = 20,
                             seed: int = 7) -> list[BinaryQuestion]:
    """Resolved YES/NO markets from the public Manifold API (no key needed)."""
    import random

    import requests

    api = "https://api.manifold.markets/v0"
    after_ms = datetime.fromisoformat(resolved_after).replace(tzinfo=timezone.utc).timestamp() * 1000
    pool: list[dict] = []
    sort = "resolve-date"  # most recently resolved first
    for offset in range(0, 1000, 100):
        params = {"term": "", "filter": "resolved", "contractType": "BINARY",
                  "sort": sort, "limit": 100, "offset": offset}
        r = requests.get(f"{api}/search-markets", timeout=30, params=params)
        if r.status_code == 400 and sort != "newest":
            logger.warning("Manifold sort %s refused, using newest", sort)
            sort = params["sort"] = "newest"
            r = requests.get(f"{api}/search-markets", timeout=30, params=params)
        if r.status_code == 400 and pool:
            break  # the API caps the offset
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        pool += [m for m in batch
                 if m.get("resolution") in ("YES", "NO")
                 and (m.get("resolutionTime") or 0) >= after_ms
                 and (m.get("createdTime") or 0) >= after_ms - 120 * 86400 * 1000
                 and (m.get("uniqueBettorCount") or 0) >= min_bettors]
    logger.info("Manifold pool: %d resolved markets", len(pool))
    random.Random(seed).shuffle(pool)
    out: list[BinaryQuestion] = []
    for m in pool[:n]:
        full = requests.get(f"{api}/market/{m['id']}", timeout=30).json()
        desc = (full.get("textDescription") or "").strip()[:3000]
        out.append(BinaryQuestion(
            question_text=m["question"],
            background_info=desc,
            resolution_criteria=desc or "Resolves YES if the event in the question happens.",
            fine_print="",
            page_url=m.get("url") or f"https://manifold.markets/{m['id']}",
            resolution_string="yes" if m["resolution"] == "YES" else "no",
        ))
    return out


async def run(n: int, models: list[str], predictions: int, out_dir: str,
              resolved_after: str, tournaments: list[str], source: str = "manifold") -> dict:
    base = build_llms()
    if base is None:
        raise SystemExit("No LLM key set")
    if source == "manifold":
        questions = await asyncio.to_thread(fetch_manifold_questions, n, resolved_after)
    else:
        questions = await asyncio.to_thread(fetch_questions, n, resolved_after, tournaments)
    logger.info("Backtest on %d resolved binary questions", len(questions))

    preds: dict[str, dict[str, float]] = {}
    errors: dict[str, int] = {}
    for model in models:
        bot = FallTemplateBot2026(
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
        logger.info("%s done in %.0f s, %d errors", model,
                    (datetime.now(timezone.utc) - started).total_seconds(), errors[model])

    rows = []
    for q in questions:
        row = {"url": q.page_url, "title": q.question_text[:120],
               "outcome": bool(q.typed_resolution), "pred": {}}
        for model in models:
            if q.page_url in preds[model]:
                row["pred"][model] = preds[model][q.page_url]
        got = list(row["pred"].values())
        if len(models) > 1 and len(got) == len(models):
            row["pred"]["ensemble_median"] = statistics.median(got)
            row["pred"]["ensemble_mean"] = statistics.mean(got)
        rows.append(row)

    variants = list(models) + (["ensemble_median", "ensemble_mean"] if len(models) > 1 else [])
    common = [r for r in rows if all(v in r["pred"] for v in variants)]
    summary = {}
    for v in variants:
        pairs = [(r["pred"][v], r["outcome"]) for r in common]
        if not pairs:
            continue
        summary[v] = {
            "answered": sum(1 for r in rows if v in r["pred"]),
            "score": statistics.mean(baseline_score(p, o) for p, o in pairs),
            "brier": statistics.mean(brier(p, o) for p, o in pairs),
            "best_logit_scale": best_scale(pairs),
            "loo_scaled_score": loo_scaled_score(pairs),
            "errors": errors.get(v, 0),
        }
    yes_rate = statistics.mean(1.0 if r["outcome"] else 0.0 for r in common) if common else None
    result = {
        "date": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "resolved_after": resolved_after,
        "tournaments": tournaments,
        "n_questions": len(questions),
        "n_common": len(common),
        "yes_rate": yes_rate,
        "predictions_per_question": predictions,
        "summary": summary,
        "rows": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "backtest.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def render(result: dict) -> str:
    lines = [
        f"BACKTEST {result['date'][:16]} source={result.get('source')} resolved_after={result['resolved_after']} "
        f"tournaments={result['tournaments']} questions={result['n_questions']} "
        f"common={result['n_common']} yes_rate={result['yes_rate']} "
        f"predictions/question={result['predictions_per_question']}",
        "50% baseline: score 0, Brier 0.25",
        "variant | answered | score (higher better) | Brier (lower better) | best_scale | LOO scaled score",
    ]
    for v, s in result["summary"].items():
        loo = s["loo_scaled_score"]
        lines.append(
            f"{v} | {s['answered']} | {s['score']:.2f} | {s['brier']:.4f} | "
            f"{s['best_logit_scale']:.2f} | {'-' if loo is None else f'{loo:.2f}'}"
        )
    lines.append("ROWS: outcome | predictions | title")
    for r in result["rows"]:
        p = " ".join(f"{k.split('/')[-1]}={v:.2f}" for k, v in r["pred"].items())
        lines.append(f"{'YES' if r['outcome'] else 'NO '} | {p} | {r['title'][:70]}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument(
        "--models",
        default="gemini/gemini-flash-lite-latest,gemini/gemini-3.1-flash-lite,gemini/gemma-4-31b-it",
    )
    ap.add_argument("--predictions", type=int, default=3)
    ap.add_argument("--resolved-after", default="2026-09-01")
    ap.add_argument("--tournaments", default="", help="Comma-separated tournament slugs or ids")
    ap.add_argument("--source", choices=["manifold", "metaculus"], default="manifold",
                    help="metaculus needs a human account token: bot tokens cannot see resolutions")
    ap.add_argument("--out", default="reports/backtest")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tournaments = [t.strip() for t in a.tournaments.split(",") if t.strip()]
    res = asyncio.run(run(a.n, models, a.predictions, a.out, a.resolved_after, tournaments, a.source))
    text = render(res)
    with open(os.path.join(a.out, "backtest.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
