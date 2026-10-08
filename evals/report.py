"""Render a benchmark run into Markdown and SVG. Every number comes from the run's JSON files.

uv run --project api python evals/report.py                 # latest run of each mode
uv run --project api python evals/report.py artifacts/runs/<run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "artifacts/runs"
RESULTS = ROOT / "docs/results"
README = ROOT / "README.md"
MARKERS = ("<!-- results:{mode}:start -->", "<!-- results:{mode}:end -->")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def overview(summary: dict[str, Any]) -> str:
    rows = []
    for name, s in summary.items():
        rows.append(
            [
                name,
                pct(s["evidence_hit_rate"]),
                pct(s["page_hit_rate"]),
                f"{s['mrr']:.3f}",
                pct(s["answer_accuracy"]),
                pct(s["citation_precision"]),
                pct(s["supported_rate"]),
                pct(s.get("judge_grounded_rate")),
                pct(s["false_refusal_rate"]),
                pct(s["correct_refusal_rate"]),
                pct(s["injection_followed_rate"]),
            ]
        )
    return table(
        [
            "Config",
            "Evidence hit@5",
            "Page hit@5",
            "MRR",
            "Answer acc.",
            "Citation prec.",
            "Supported",
            "Judge grounded",
            "False refusal",
            "Correct refusal",
            "Injection followed",
        ],
        rows,
    )


def categories(summary: dict[str, Any]) -> str:
    names = sorted({c for s in summary.values() for c in s["by_category"]} - {"unanswerable"})
    rows = []
    for config, s in summary.items():
        cells = []
        for c in names:
            cat = s["by_category"].get(c)
            if cat is None:
                cells.append("n/a")
                continue
            hit, acc = pct(cat["evidence_hit_rate"]), pct(cat["answer_accuracy"])
            cells.append(f"{hit} / {acc} (n={cat['answerable']})")
        rows.append([config, *cells])
    return table(["Config", *names], rows)


def failures(summary: dict[str, Any]) -> str:
    kinds = [
        "parse_miss",
        "retrieval_miss",
        "false_refusal",
        "wrong_answer",
        "wrong_citation",
        "injection_followed",
        "missed_refusal",
    ]
    rows = [[name, *[str(s["failures"].get(k, 0)) for k in kinds]] for name, s in summary.items()]
    return table(["Config", *kinds], rows)


def latency(summary: dict[str, Any]) -> str:
    rows = []
    for name, s in summary.items():
        lat = s["latency_ms"]
        usage = s["usage"]
        rows.append(
            [
                name,
                *(
                    _p50_p95(lat.get(stage))
                    for stage in ("retrieve", "generate", "rewrite", "total")
                ),
                f"{usage['input_tokens']:,} / {usage['output_tokens']:,} ({usage['calls']} calls)",
            ]
        )
    return table(
        [
            "Config",
            "Retrieve p50/p95 ms",
            "Generate p50/p95 ms",
            "Rewrite p50/p95 ms",
            "Total p50/p95 ms",
            "Answer tokens in/out",
        ],
        rows,
    )


def _swap(env: dict[str, Any], key: str) -> str:
    state = env.get(key) or {}
    swap = state.get("swap") or ""
    used = swap.split("used = ")[1].split()[0] if "used = " in swap else "not recorded"
    return used


def _p50_p95(stage: dict[str, Any] | None) -> str:
    return f"{ms(stage['p50'])} / {ms(stage['p95'])}" if stage else "n/a"


def environment(env: dict[str, Any]) -> str:
    pkgs = ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)
    ingest = env.get("ingest") or {}
    pages = sum(d["pages"] for d in ingest.values())
    ocr = sum(d["ocr_pages"] for d in ingest.values())
    parse = sum(d["timings_ms"].get("parse", 0) for d in ingest.values())
    lines = [
        f"- Run: `{env['run_id']}`, commit `{env['git_commit'][:10]}`"
        + (" (uncommitted changes)" if env["git_dirty"] else ""),
        f"- Machine: {env['cpu']}, {env['cpu_count']} cores, {env['memory_gb']} GB RAM, "
        f"{env['platform']}",
        f"- Swap at start / end: {_swap(env, 'memory_at_start')} / {_swap(env, 'memory_at_end')}",
        f"- Python {env['python']}; {env['tesseract']}; {pkgs}",
        f"- Answer model: `{env['answer_model']}` (prompt `{env['prompt_version']}`)"
        + (
            f"; judge `{env['judge_model']}` (`{env['judge_version']}`)"
            if env.get("judge_model")
            else ""
        ),
        f"- Embeddings `{env['embedding_model']}`; reranker `{env['reranker_model']}`",
        f"- Questions: split `{env['split']}`; {env['repeats']} run(s) per question for latency",
    ]
    if ingest:
        lines.append(
            f"- Ingest in this run: {len(ingest)} documents, {pages} pages ({ocr} OCR), "
            f"parse time {parse / 1000:.1f} s total"
        )
    return "\n".join(lines)


def chart(summary: dict[str, Any]) -> str:
    """Grouped bar chart of evidence hit rate per category, one bar per config."""
    cats = sorted({c for s in summary.values() for c in s["by_category"]} - {"unanswerable"})
    configs = list(summary)
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d"]
    bar, gap, height, left, top = 14, 22, 220, 50, 20
    group = bar * len(configs) + gap
    width = left + group * len(cats) + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 110}" '
        f'font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + height * (1 - tick)
        parts.append(f'<line x1="{left}" x2="{width - 10}" y1="{y}" y2="{y}" stroke="#ddd"/>')
        parts.append(
            f'<text x="{left - 6}" y="{y + 4}" text-anchor="end">{int(tick * 100)}%</text>'
        )
    for ci, cat in enumerate(cats):
        x0 = left + ci * group + gap / 2
        for i, config in enumerate(configs):
            value = (summary[config]["by_category"].get(cat) or {}).get("evidence_hit_rate") or 0
            h = height * value
            parts.append(
                f'<rect x="{x0 + i * bar}" y="{top + height - h}" width="{bar - 2}" '
                f'height="{h}" fill="{palette[i % len(palette)]}"><title>{config} {cat}: '
                f"{pct(value)}</title></rect>"
            )
        parts.append(
            f'<text x="{x0 + bar * len(configs) / 2}" y="{top + height + 14}" '
            f'text-anchor="middle">{cat}</text>'
        )
    for i, config in enumerate(configs):
        y = top + height + 34 + i * 15
        parts.append(
            f'<rect x="{left}" y="{y - 9}" width="10" height="10" '
            f'fill="{palette[i % len(palette)]}"/>'
        )
        parts.append(f'<text x="{left + 15}" y="{y}">{config}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render(run: Path) -> tuple[str, str]:
    summary = json.loads((run / "summary.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    env["run_id"] = run.name
    mode = env["llm"]
    svg_name = f"evidence-by-category-{mode}.svg"
    (RESULTS / svg_name).write_text(chart(summary))
    body = "\n\n".join(
        [
            f"# Benchmark results: `{mode}`",
            "Generated by `evals/report.py` from `artifacts/runs/"
            + run.name
            + "`. Do not edit by hand.",
            "## Environment",
            environment(env),
            "## Overall (answerable questions unless noted)",
            overview(summary),
            "## Evidence hit@5 / answer accuracy by category",
            categories(summary),
            f"![Evidence hit rate by category]({svg_name})",
            "## Failure categories (count of questions)",
            failures(summary),
            "## Latency and tokens",
            latency(summary),
            "Metric definitions: [docs/evaluation.md](../evaluation.md).",
        ]
    )
    (RESULTS / f"latest-{mode}.md").write_text(body + "\n")
    short = "\n\n".join(
        [
            overview(summary),
            f"Run `{run.name}` on {env['cpu']} ({env['memory_gb']} GB). "
            f"Full tables: [docs/results/latest-{mode}.md](docs/results/latest-{mode}.md).",
        ]
    )
    return mode, short


def update_readme(mode: str, block: str) -> None:
    if not README.exists():
        return
    start, end = (m.format(mode=mode) for m in MARKERS)
    text = README.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        README.write_text(f"{head}{start}\n{block}\n{end}{tail}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        runs = [Path(sys.argv[1])]
    else:
        complete = sorted(p for p in RUNS.glob("*") if (p / "summary.json").exists())
        latest: dict[str, Path] = {}
        for p in complete:
            latest[p.name.rsplit("-", 1)[-1]] = p
        runs = list(latest.values())
    for run in runs:
        mode, block = render(run)
        update_readme(mode, block)
        print(f"rendered {run.name}")


if __name__ == "__main__":
    main()
