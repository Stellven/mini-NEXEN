from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import db
from .agents import SupervisorAgent
from .config import PLANS_DIR, ensure_dirs
from .llm import build_client, load_llm_config
from .skills_runtime import SkillContext, build_default_runner


@dataclass
class ResearchResult:
    plan_path: Path
    plan_markdown: str


def _plan_filename(now: datetime) -> str:
    return now.strftime("%Y_%m_%d_%H_%M_plan.md")


def run_research(
    topic: str,
    rounds: int,
    top_k: int,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    discover_model: bool | None = None,
) -> ResearchResult:
    ensure_dirs()
    db.init_db()

    # Record the topic as a user interest for continuity.
    db.add_interest(topic=topic, notes="auto: research query")

    runner = build_default_runner()
    supervisor = SupervisorAgent(runner)

    llm_config = load_llm_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        discover_model=discover_model,
    )
    llm_client = build_client(llm_config)

    ctx = SkillContext(topic=topic, max_rounds=rounds, top_k=top_k, llm=llm_client)
    ctx = supervisor.run(ctx)

    plan_md = ctx.plan_md
    now = datetime.now()
    plan_path = PLANS_DIR / _plan_filename(now)
    plan_path.write_text(plan_md, encoding="utf-8")

    return ResearchResult(plan_path=plan_path, plan_markdown=plan_md)
