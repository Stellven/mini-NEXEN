from __future__ import annotations

from dataclasses import dataclass

from .llm import log_task_event
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
        self.retriever = Agent(
            "retriever",
            [
                "infer_query",
                "collect_interests",
                "collect_methods",
                "load_profile",
                "apply_skill_hints",
                "systems-engineering",
                "web_retrieve",
                "detect_contradictions",
                "retrieve_subgraph",
            ],
        )
        self.planner = Agent("planner", ["plan_research", "refine_plan"])
        self.outliner = Agent("outliner", ["build_outline", "persist_plan"])

    def run(self, ctx: SkillContext) -> SkillContext:
        max_rounds = max(1, ctx.max_rounds)
        for idx in range(max_rounds):
            ctx.round_number = idx + 1
            ctx.kg_updated = False
            log_task_event(f"=== Round {ctx.round_number}/{max_rounds} ===")
            ctx = self.retriever.run(ctx, self.runner)
            ctx = self.planner.run(ctx, self.runner)
            if ctx.plan and ctx.plan.readiness == "ready":
                break
        ctx = self.outliner.run(ctx, self.runner)
        return ctx
