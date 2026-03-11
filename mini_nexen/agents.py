from __future__ import annotations

from dataclasses import dataclass

from .llm import clear_log_context, log_task_event, set_log_context
from .planning import validate_outline, validate_plan
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
        entry_point = "planner"
        for idx in range(max_rounds):
            ctx.round_number = idx + 1
            ctx.kg_updated = False
            ctx.web_rounds_used = 0
            ctx.web_last_added = 0
            ctx.web_stop_early = False
            ctx.web_stop_reason = ""
            set_log_context(external_round=ctx.round_number, external_total=max_rounds, component="Orchestrator")
            log_task_event(f"Pipeline loop {ctx.round_number}/{max_rounds} start")

            if entry_point == "planner":
                ctx = self.planner.run(ctx, self.runner)
            if entry_point in {"planner", "retriever"}:
                retrieval_rounds = ctx.web_max_rounds if ctx.web_enabled else 1
                for _ in range(max(1, retrieval_rounds)):
                    ctx = self.retriever.run(ctx, self.runner)
                    if ctx.web_stop_reason == "sufficient":
                        break
                    if ctx.web_stop_early and not ctx.web_forced:
                        break

            ctx = self.outliner.run(ctx, self.runner)

            action = (ctx.outline_review_action or "").strip().lower()
            if action == "retrieve_more":
                entry_point = "retriever"
                log_task_event("Outline review requested more retrieval; restarting pipeline at retriever.")
                continue
            if action == "plan_retry":
                entry_point = "planner"
                ctx.plan_revision_feedback = ctx.outline_plan_gaps + ctx.outline_review_feedback
                log_task_event("Outline review requested planner retry; restarting pipeline at planner.")
                continue
            if action == "outline_retry":
                entry_point = "outliner"
                log_task_event("Outline review requested outline retry; rerunning outliner.")
                continue
            if ctx.outline_validation and not bool(ctx.outline_validation.get("ok")):
                entry_point = "planner"
                log_task_event("Outline validation failed; restarting pipeline at planner.")
                continue
            break
        plan_status = "missing"
        plan_ok = None
        if ctx.plan:
            plan_status = ctx.plan.readiness
            plan_validation = ctx.plan_validation or validate_plan(ctx.plan)
            plan_ok = bool(plan_validation.get("ok"))
        outline_ok = None
        if ctx.outline:
            outline_validation = ctx.outline_validation or validate_outline(
                ctx.outline, ctx.output_language, kg_fact_cards=ctx.kg_fact_cards
            )
            outline_ok = bool(outline_validation.get("ok"))
        retrieval_status = (
            f"enabled={ctx.web_enabled} auto={ctx.web_auto} rounds={ctx.web_rounds_used}/{ctx.web_max_rounds} "
            f"last_added={ctx.web_last_added} stop_reason={ctx.web_stop_reason or 'none'}"
        )
        log_task_event(f"Final status | plan: {plan_status} ok={plan_ok}")
        log_task_event(f"Final status | retrieval: {retrieval_status}")
        log_task_event(f"Final status | outline: ok={outline_ok} action={ctx.outline_review_action or 'none'}")
        clear_log_context(component=True)
        return ctx
