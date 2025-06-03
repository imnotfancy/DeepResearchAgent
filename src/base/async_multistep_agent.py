import importlib
import inspect
import json
import json5
import os
import re
import tempfile
import textwrap
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypedDict, Union

import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from dataclasses import dataclass # Import dataclass
from src.tools.default_tools import TOOL_MAPPING
from src.tools.final_answer import FinalAnswerTool
from src.tools.executor.local_python_executor import BASE_BUILTIN_MODULES
from src.memory import (ActionStep, # Already here
                        AgentMemory, # Already here
                        FinalAnswerStep, # Already here
                        PlanningStep, # Already here
                        SystemPromptStep, # Already here
                        UserPromptStep, # Already here
                        TaskStep, # Already here
                        Message, # Already here
                        ToolCall) # Import ToolCall from src.memory
from src.models import (
    ChatMessage, # Already here
    MessageRole, # Already here
    Model
)
from src.logger import (
    AgentLogger,
    LogLevel,
    Monitor,
)

from src.tools import AsyncTool
from src.exception import (
    AgentError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError, # Import AgentToolCallError
    AgentToolExecutionError, # Import AgentToolExecutionError
)

from src.utils import (
    is_valid_name,
    make_init_file,
    truncate_content,
    # parse_json_if_needed # This will be defined locally for now
)

from src.logger import logger
# from src.config import config # Config is now passed in __init__

# ToolCall is now imported from src.memory
# Keep parse_json_if_needed here for now
def parse_json_if_needed(text: str) -> Any:
    """
    Tries to parse the text as JSON. If successful, returns the parsed object.
    If the text is not valid JSON or another error occurs, returns the original text.
    Handles cases where the text might be a JSON string within a larger string (e.g. ```json...```)
    """
    # Strip common code block markers
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json5.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text # Return original text if not valid JSON or not a string/bytes

def get_variable_names(self, template: str) -> Set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)

    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        user_prompt (`str`): User prompt.
        task_instruction (`str`): Task instruction.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    user_prompt: str
    task_instruction: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    user_prompt="",
    task_instruction="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


class AsyncMultiStepAgent(ABC):
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: list[AsyncTool],
        model: Model,
        config: Any, # TODO: Add type hint for config
        prompt_templates: PromptTemplates | None = None,
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
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.config = config

        if prompt_templates is not None:
            self.prompt_templates = prompt_templates
        elif self.config and hasattr(self.config, "template_path"):
            # Load prompt templates from config template_path
            project_root = Path(__file__).resolve().parents[2]
            full_template_path = project_root / self.config.template_path
            with open(full_template_path, "r") as f:
                self.prompt_templates = yaml.safe_load(f)
        else:
            self.prompt_templates = EMPTY_PROMPT_TEMPLATES

        if self.prompt_templates is not None and self.prompt_templates is not EMPTY_PROMPT_TEMPLATES :
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(self.prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in self.prompt_templates.keys() and (subkey in self.prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools) # self.tools is initialized here
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None # self.task is initialized before calling initialize_task_instruction

        self.system_prompt = self.initialize_system_prompt()
        self.user_prompt = self.initialize_user_prompt()
        self.memory = AgentMemory(system_prompt=self.system_prompt, user_prompt=self.user_prompt)

        self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, AsyncTool) for tool in tools), "All elements must be instance of AsyncTool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        
        # Initialize the task instruction
        self.task = self.initialize_task_instruction()
        
        self.interrupt_switch = False
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        self.user_prompt = self.initialize_user_prompt()
        self.memory.user_prompt = UserPromptStep(user_prompt=self.user_prompt)
        
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            return await self._run(task=self.task, max_steps=max_steps, images=images)

        step_queue = deque(maxlen=1)
        async for step in self._run(task=self.task, max_steps=max_steps, images=images):
            step_queue.append(step)

        return step_queue[0].final_answer

    async def _run(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ):
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)
            step_start_time = time.time()
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_step = await self._generate_planning_step(
                    task, is_first_step=(self.step_number == 1), step=self.step_number
                )
                self.memory.steps.append(planning_step)
                yield planning_step
            action_step = ActionStep(
                step_number=self.step_number, start_time=step_start_time, observations_images=images
            )
            try:
                final_answer = await self._execute_step(task, action_step)
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step, step_start_time)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if final_answer is None and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task, images, step_start_time)
            yield action_step
        yield FinalAnswerStep(final_answer)

    async def _execute_step(self, task: str, memory_step: ActionStep) -> None | Any:
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        final_answer = await self.step(memory_step)
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)
        return final_answer

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - step_start_time
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"], step_start_time: float) -> Any:
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number, error=AgentMaxStepsError("Reached max steps.", self.logger)
        )
        final_memory_step.action_output = final_answer
        final_memory_step.end_time = time.time()
        final_memory_step.duration = final_memory_step.end_time - step_start_time
        self.memory.steps.append(final_memory_step)
        for callback in self.step_callbacks:
            callback(final_memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                final_memory_step, agent=self
            )
        return final_answer

    async def _generate_planning_step(self, task, is_first_step: bool, step: int) -> PlanningStep:
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                }
            ]
            plan_message = self.model(input_messages, stop_sequences=["<end_plan>"])
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message.content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = await self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
            plan_update_post = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            plan_message = self.model(input_messages, stop_sequences=["<end_plan>"])
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message.content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        return PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=plan_message,
        )

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self) -> str:
        """Initializes the system prompt using the prompt_templates."""
        return populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
            },
        )

    def initialize_user_prompt(self) -> str:
        """Initializes the user prompt using the prompt_templates."""
        return populate_template(
            self.prompt_templates["user_prompt"],
            variables={
                "task": self.task,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
            },
        )

    def initialize_task_instruction(self) -> str:
        """Initializes the task instruction using the prompt_templates."""
        # Ensure self.task is the raw task string before populating
        if not self.task:
            raise ValueError("Task must be set before initializing task instruction.")
        return populate_template(
            self.prompt_templates["task_instruction"],
            variables={"task": self.task},
        )

    def _substitute_state_variables(self, text: str) -> str:
        """Substitutes state variables in the given text.
        Example: "Hello {{name}}" with state {"name": "World"} -> "Hello World"
        """
        if not isinstance(text, str):
            return text

        # Find all placeholders like {{variable_name}}
        placeholders = re.findall(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", text)

        substituted_text = text
        for placeholder in placeholders:
            if placeholder in self.state:
                substituted_text = substituted_text.replace(f"{{{{{placeholder}}}}}", str(self.state[placeholder]))
            else:
                self.logger.log(f"Warning: Variable '{placeholder}' not found in agent state.", LogLevel.WARNING)
        return substituted_text

    async def execute_tool_call(self, tool_call: "ToolCall") -> Any:
        """Executes a tool call and returns the output.

        Args:
            tool_call (ToolCall): The tool call to execute.

        Returns:
            Any: The output of the tool.
        """
        tool_name = tool_call.name # Use tool_call.name
        tool_arguments = tool_call.arguments # Use tool_call.arguments

        # Substitute state variables in tool arguments
        if isinstance(tool_arguments, str):
            tool_arguments = self._substitute_state_variables(tool_arguments)
        elif isinstance(tool_arguments, dict):
            tool_arguments = {
                k: self._substitute_state_variables(v) if isinstance(v, str) else v
                for k, v in tool_arguments.items()
            }

        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                # Log the tool call before execution
                self.logger.log(
                    f"Calling tool {tool_name} with parameters: {tool_arguments}",
                    level=LogLevel.INFO,
                    color="yellow"
                )
                tool_output = await tool.run(tool_arguments, agent=self) # Pass agent instance
                # Log the tool output after execution
                self.logger.log(
                    f"Tool {tool_name} output: {tool_output}",
                    level=LogLevel.INFO,
                    color="yellow"
                )
                return tool_output
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {e}"
                self.logger.log(error_message, LogLevel.ERROR, color="red")
                raise AgentToolExecutionError(error_message, self.logger) from e
        elif tool_name in self.managed_agents:
            managed_agent = self.managed_agents[tool_name]
            try:
                # Log the managed agent call before execution
                self.logger.log(
                    f"Calling managed agent {tool_name} with task: {tool_arguments}",
                    level=LogLevel.INFO,
                    color="yellow"
                )
                # Ensure tool_arguments is a string (task for the managed agent)
                if not isinstance(tool_arguments, str):
                    tool_arguments = str(tool_arguments)

                agent_output = await managed_agent(task=tool_arguments) # Pass task to managed_agent
                # Log the managed agent output after execution
                self.logger.log(
                    f"Managed agent {tool_name} output: {agent_output}",
                    level=LogLevel.INFO,
                    color="yellow"
                )
                return agent_output
            except Exception as e:
                error_message = f"Error executing managed agent {tool_name}: {e}"
                self.logger.log(error_message, LogLevel.ERROR, color="red")
                raise AgentToolExecutionError(error_message, self.logger) from e
        else:
            error_message = f"Tool or managed agent '{tool_name}' not found."
            self.logger.log(error_message, LogLevel.ERROR, color="red")
            raise AgentToolCallError(error_message, self.logger)

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    async def write_memory_to_messages(
        self,
        summary_mode: bool | None = False,
    ) -> list[Message]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        messages.extend(self.memory.user_prompt.to_messages(summary_mode=summary_mode))
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    async def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += await self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    @abstractmethod
    async def step(self, memory_step: ActionStep) -> None | Any:
        """Executes a single step of the agent's thought-action-observation loop.

        Args:
            memory_step (ActionStep): The memory step to record the action and observation.

        Returns:
            None | Any: The final answer if the agent decides to stop, otherwise None.
        """
        try:
            # 1. Generate action from the model
            messages = await self.write_memory_to_messages()
            # Add any images to the last message if available in the current memory_step
            if memory_step.observations_images:
                if messages and messages[-1]["role"] == MessageRole.USER: # Add to last user message
                    if isinstance(messages[-1]["content"], str): # if content is string, convert to list
                         messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
                    messages[-1]["content"].extend([{"type": "image"} for _ in memory_step.observations_images])
                else: # Or create a new user message for images
                    messages.append(ChatMessage(role=MessageRole.USER, content=[{"type": "image"} for _ in memory_step.observations_images]))

            model_output: ChatMessage = self.model(
                messages,
                images=memory_step.observations_images, # Pass images to the model
                grammar=self.grammar, # Pass grammar if any
            )
            memory_step.model_input_messages = messages
            memory_step.model_output_message = model_output

            # TODO: This parsing logic might need to be made abstract or configurable
            # For now, assuming a simple split logic, but this is a common source of variation.
            # If your agent uses JSON or XML, you'll need to override this.
            try:
                # Attempt to parse JSON if applicable (common for tool use)
                parsed_output = parse_json_if_needed(model_output.content)
                # Using src.memory.ToolCall: requires 'name', 'arguments', 'id'. 'rationale' is not part of it.
                # 'id' will be tricky to generate here if not provided by the model directly in this format.
                # For now, if the model returns tool_name/tool_params, we adapt.
                # A more robust solution might involve the model returning a structure closer to src.memory.ToolCall.

                rationale = parsed_output.get("rationale") if isinstance(parsed_output, dict) else None
                if rationale:
                    memory_step.rationale = rationale

                if isinstance(parsed_output, dict) and "tool_name" in parsed_output and "tool_params" in parsed_output:
                     # Create a src.memory.ToolCall object. 'id' might be missing or need a placeholder.
                     # The model might not provide an 'id' in this custom format.
                     # This is a gap: src.memory.ToolCall expects an 'id'.
                     # For now, let's use a placeholder id if not present.
                     tool_id = parsed_output.get("id", "placeholder_id")
                     tool_call = ToolCall(name=parsed_output["tool_name"], arguments=parsed_output["tool_params"], id=tool_id)
                elif isinstance(parsed_output, dict) and "final_answer" in parsed_output: # Check for final answer
                    final_answer = parsed_output["final_answer"]
                    memory_step.action_output = final_answer
                    memory_step.rationale = rationale if rationale else parsed_output.get("rationale", "No rationale provided for final answer.")
                    self.logger.log_final_answer(final_answer, self.name)
                    return final_answer
                else: # Fallback or if no specific parsing needed, treat as simple action.
                    # This part needs to be robust. What if there's no split token?
                    # This is a placeholder for more sophisticated action parsing.
                    # For now, we assume a simple "final_answer" or tool call structure.
                    # If it's not a structured call, it might be a direct answer or need other parsing.
                    # This part might need to be made abstract if agents vary too much here.
                    # For now, if not a tool call, assume it could be a final answer if it's a string.
                    if isinstance(model_output.content, str) and not self.tools: # No tools, might be direct answer
                         self.logger.log_final_answer(model_output.content, self.name)
                         return model_output.content
                    raise AgentParsingError(f"Could not parse LLM output: {model_output.content}", self.logger)

            except AgentParsingError: # Fallback to simpler split if JSON parsing fails or isn't applicable.
                 # This is a common pattern: Rationale <split_token> Action
                 # The split_token should ideally be defined in config or prompts.
                 # Using a default or expecting it from prompts.
                 # This part is highly dependent on the expected output format from the LLM.
                split_token = getattr(self.config, "action_split_token", "Tool Call:") # Example token
                rationale, action_str = self.extract_action(model_output.content, split_token)
                memory_step.rationale = rationale
                # Further parsing of action_str to ToolCall might be needed here.
                # This is a placeholder and needs to be robust.
                # Example: tool_name(tool_params_json_string)
                match = re.match(r"(\w+)\((.*)\)", action_str)
                if match:
                    tool_name, tool_params_str = match.groups()
                    try:
                        tool_arguments = json5.loads(tool_params_str) # Use json5 for more flexible parsing
                    except json.JSONDecodeError:
                        # if params are not json, pass as string
                        tool_arguments = tool_params_str
                    # Again, src.memory.ToolCall needs an 'id'.
                    tool_call = ToolCall(name=tool_name.strip(), arguments=tool_arguments, id="placeholder_id_fallback")
                else: # If no clear tool call format, could be final answer or error
                    if "final_answer" in action_str.lower() or not self.tools: # Heuristic for final answer
                        final_answer_text = action_str.replace("Final Answer:", "").strip()
                        memory_step.rationale = rationale if rationale else "No rationale provided."
                        memory_step.action_output = final_answer_text
                        self.logger.log_final_answer(final_answer_text, self.name)
                        return final_answer_text
                    raise AgentParsingError(f"Could not parse action string: {action_str}", self.logger)

            # memory_step.action = tool_call # 'action' field does not exist on ActionStep, tool_calls is used.
            memory_step.tool_calls = [tool_call] # Store the tool call in the memory step

            # 2. Execute tool call
            if tool_call.name == "final_answer": # Use tool_call.name
                # This is often a special tool that signals the end.
                # The parameters to final_answer usually contain the actual response.
                final_answer_content = tool_call.arguments.get("answer") if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                if final_answer_content is None:
                    raise AgentToolExecutionError("final_answer tool called without 'answer' parameter.", self.logger)

                memory_step.action_output = final_answer_content # Keep action_output for final answer
                self.logger.log_final_answer(final_answer_content, self.name)
                return final_answer_content

            observation = await self.execute_tool_call(tool_call)
            memory_step.observation = observation
            memory_step.action_output = observation # For consistency, action_output can be the observation

            # Store observation in state if tool indicates it (e.g. via a special return or config)
            # This part is application-specific. Example:
            # Using tool_call.name and tool_call.arguments
            if hasattr(self.tools.get(tool_call.name), "store_output_in_state") and self.tools[tool_call.name].store_output_in_state:
                 if isinstance(observation, dict): # if observation is dict, update state
                      self.state.update(observation)
                 elif isinstance(observation, str) and isinstance(tool_call.arguments, dict) and tool_call.arguments.get("variable_name"):
                      self.state[tool_call.arguments["variable_name"]] = observation
                 # else, how to name it in state? Maybe tool_name_output?
                 # self.state[f"{tool_call.name}_output"] = observation


        except AgentToolCallError as e: # Tool name validation error
            memory_step.error = e
            memory_step.observation = str(e)
        except AgentToolExecutionError as e: # Tool execution error
            memory_step.error = e
            memory_step.observation = str(e)
        except AgentParsingError as e: # LLM output parsing error
            memory_step.error = e
            memory_step.observation = str(e) # Observation can be the error message for the LLM to correct
        except AgentGenerationError as e: # LLM generation error (e.g. API error)
            # This is more critical and might indicate issues with the LLM service
            memory_step.error = e
            # Unlike other errors, this might not be something the agent can easily recover from by itself.
            # Depending on severity, might re-raise or try a recovery mechanism.
            raise # Re-raise for now, as it's a generation infrastructure problem
        except Exception as e: # Catch any other unexpected error during the step
            unexpected_error = AgentError(f"Unexpected error during agent step: {e}", self.logger)
            memory_step.error = unexpected_error
            memory_step.observation = str(unexpected_error)

        return None # No final answer yet, continue stepping

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    async def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        report = await self.run(full_task, **kwargs)
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in await self.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4, ensure_ascii=False)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """).strip()
        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        # Render the app.py file from Jinja2 template
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = getattr(importlib.import_module("smolagents.models"), model_info["class"])
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(AsyncTool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            managed_agent_class = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(managed_agent_class.from_dict(agent_dict["managed_agents"][managed_agent_name]))
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "grammar": agent_dict.get("grammar"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )