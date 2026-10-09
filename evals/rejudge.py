"""Re-run the groundedness judge on an existing run's answers, without regenerating them.

    uv run --project api python evals/rejudge.py artifacts/runs/<run_id> --max-cost-usd 1

Cited chunk text is read back from the evaluation database by chunk id, so the database must
still hold the ingest used by the run. The previous records are kept as
records.<old judge version>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from intellidoc.answer.models import Source, make_chat
from intellidoc.evaluation.judge import JUDGE_VERSION, ChatJudge
from intellidoc.settings import get_settings

sys.path.insert(0, str(Path(__file__).parent))
from run import ROOT, cost_usd, load_configs, load_prices, summarise_records


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--max-cost-usd", type=float, required=True)
    args = parser.parse_args()
    run = Path(args.run)
    env = json.loads((run / "environment.json").read_text())
    old_version = env.get("judge_version") or "judge-v1"
    if old_version == JUDGE_VERSION:
        raise SystemExit(f"{run} was already judged with {JUDGE_VERSION}")

    settings = get_settings()
    provider = env["llm"]
    judge = ChatJudge(
        make_chat(provider, settings.judge_model, settings.api_key(provider)), settings.judge_model
    )
    prices = load_prices()
    records = [json.loads(line) for line in (run / "records.jsonl").open()]
    url = os.environ.get("EVAL_DATABASE_URL", settings.database_url)

    spent = 0.0
    with psycopg.connect(url) as conn:
        for r in records:
            if not r["judge"] and not r["judge_usage"]["calls"]:
                continue  # refused answers are not judged
            ids = [c["chunk_id"] for c in r["citations"]]
            rows = {
                row[0]: row[1:]
                for row in conn.execute(
                    "SELECT c.id, d.title, c.pages, c.text FROM chunks c "
                    "JOIN documents d ON d.id = c.doc_id WHERE c.id = ANY(%s)",
                    (ids,),
                ).fetchall()
            }
            if len(rows) != len(ids):
                raise SystemExit(
                    f"chunks missing for {r['question']['id']}; database was re-ingested"
                )
            sources = [Source(f"S{i + 1}", *rows[cid]) for i, cid in enumerate(ids)]
            if spent + 0.01 > args.max_cost_usd:
                raise SystemExit(f"budget stop at ${spent:.4f}; nothing written")
            verdict, usage = judge.judge(
                r["question"]["question"], r["answer"], sources, [run.name, r["config"], "rejudge"]
            )
            new_cost = cost_usd(usage.__dict__, judge.name, prices)
            r["cost_usd"] = round(r["cost_usd"] + new_cost, 6)  # both judge passes were paid for
            r["judge_previous"] = {"version": old_version, "verdict": r["judge"]}
            r["judge"] = verdict.model_dump() if verdict else None
            r["judge_usage"] = usage.__dict__
            spent += new_cost

    (run / f"records.{old_version}.jsonl").write_text((run / "records.jsonl").read_text())
    with (run / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    names = list(dict.fromkeys(r["config"] for r in records))
    configs = [c for c in load_configs(None) if c["name"] in names]
    (run / "summary.json").write_text(
        json.dumps(summarise_records(records, configs), indent=2) + "\n"
    )
    env["judge_version"] = JUDGE_VERSION
    env.setdefault("rejudged", []).append({"from": old_version, "cost_usd": round(spent, 4)})
    env["cost_usd_total"] = round(sum(r.get("cost_usd", 0.0) for r in records), 4)
    (run / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    total = env["cost_usd_total"]
    print(f"rejudged {run.name} with {JUDGE_VERSION}; cost ${spent:.4f}; run total ${total}")


if __name__ == "__main__":
    main()
