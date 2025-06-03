from typing import (
    Callable,
    Optional
)

from src.tools import AsyncTool
from src.base.async_multistep_agent import (AsyncMultiStepAgent, PromptTemplates) # Removed populate_template
# Removed ActionStep, ToolCall, AgentMemory as they are handled by base or not used here
from src.logger import (LogLevel, logger) # Removed YELLOW_HEX
from src.models import Model # Removed parse_json_if_needed, ChatMessage
# Removed AgentAudio, AgentImage as they were used in the 'step' method
from src.registry import register_agent
# Removed assemble_project_path, yaml, json, Panel, Text
# Removed AgentGenerationError, AgentParsingError, AgentToolExecutionError, AgentToolCallError

@register_agent("browser_use_agent")
class BrowserUseAgent(AsyncMultiStepAgent):
    def __init__(
        self,
        config, # Added config here
        tools: list[AsyncTool],
        model: Model,
        # prompt_templates is removed as it's handled by base via config
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
        # self.config = config # Removed, handled by base if needed, or passed directly

        super().__init__( # Changed to direct super().__init__
            config=config, # Pass config to base
            tools=tools,
            model=model,
            # prompt_templates is removed
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
            **kwargs # Pass any other relevant kwargs
        )

        # Removed template loading logic
        # Removed self.system_prompt initialization
        # Removed self.user_prompt initialization
        # Removed self.memory initialization

    # Removed initialize_system_prompt
    # Removed initialize_user_prompt
    # Removed initialize_task_instruction
    # Removed _substitute_state_variables
    # Removed execute_tool_call
    # Removed step