from __future__ import annotations

from dataclasses import dataclass

from .llm import clear_log_context, log_task_event, set_log_context
from .skills_runtime import SkillContext, SkillRunner


@dataclass
class Agent:
    name: str
    skills: list[str]

    def run(self, ctx: SkillContext, runner: SkillRunner) -> SkillContext:
        for skill_name in self.skills:
            ctx = runner.run(skill_name, ctx)
        return ctx


class Orchestrator:
    def __init__(self, runner: SkillRunner):
        self.runner = runner
        self.preplanner = Agent(
            "preplanner",
            [
                "infer_query",
                "collect_interests",
                "collect_methods",
                "load_profile",
                "apply_skill_hints",
                "systems-engineering",
                "plan_research",
            ],
        )
        self.planner = Agent("planner", ["plan_research"])
        self.retriever = Agent(
            "retriever",
            [
                "web_retrieve",
                "detect_contradictions",
                "retrieve_subgraph",
            ],
        )
        self.outliner = Agent("outliner", ["build_outline", "persist_plan"])

    def run(self, ctx: SkillContext) -> SkillContext:
        max_rounds = max(1, ctx.max_rounds)
        ctx = self.preplanner.run(ctx, self.runner)
        for idx in range(max_rounds):
            ctx.round_number = idx + 1
            ctx.kg_updated = False
            set_log_context(external_round=ctx.round_number, external_total=max_rounds, component="Orchestrator")
            log_task_event(f"=== Round {ctx.round_number}/{max_rounds} ===")
            ctx = self.retriever.run(ctx, self.runner)
        ctx = self.outliner.run(ctx, self.runner)
        for _ in range(2):
            action = (ctx.outline_review_action or "").strip().lower()
            if action == "retrieve_more" and not ctx.outline_triggered_retrieval:
                ctx.outline_triggered_retrieval = True
                ctx.round_number = max_rounds + 1
                ctx.kg_updated = False
                set_log_context(external_round=ctx.round_number, external_total=max_rounds + 1, component="Orchestrator")
                log_task_event("Outline review triggered retrieval rerun.")
                ctx = self.retriever.run(ctx, self.runner)
                ctx = self.outliner.run(ctx, self.runner)
                continue
            if action == "plan_retry" and not ctx.outline_triggered_plan_retry:
                ctx.outline_triggered_plan_retry = True
                ctx.plan_revision_feedback = ctx.outline_plan_gaps + ctx.outline_review_feedback
                log_task_event("Outline review triggered planner rerun.")
                ctx = self.planner.run(ctx, self.runner)
                ctx.round_number = max_rounds + 1
                ctx.kg_updated = False
                set_log_context(external_round=ctx.round_number, external_total=max_rounds + 1, component="Orchestrator")
                ctx = self.retriever.run(ctx, self.runner)
                ctx = self.outliner.run(ctx, self.runner)
                continue
            break
        clear_log_context(component=True)
        return ctx
