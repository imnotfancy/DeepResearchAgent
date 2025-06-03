from typing import (
    Callable,
    Optional
)

from src.tools import AsyncTool
from src.base.async_multistep_agent import (AsyncMultiStepAgent, PromptTemplates)
from src.logger import (LogLevel, logger)
from src.models import Model
from src.registry import register_agent

@register_agent("planning_agent")
class PlanningAgent(AsyncMultiStepAgent):
    def __init__(
        self,
        config,
        tools: list[AsyncTool],
        model: Model,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            tools=tools,
            model=model,
            max_steps=max_steps,
            add_base_tools=add_base_tools,
            verbosity_level=verbosity_level,
            grammar= grammar,
            managed_agents=managed_agents,
            step_callbacks=step_callbacks,
            planning_interval=planning_interval,
            name=name,
            description=description,
            provide_run_summary=provide_run_summary,
            final_answer_checks=final_answer_checks,
            **kwargs
        )