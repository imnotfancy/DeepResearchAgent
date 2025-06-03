# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
# import inspect # Not used
# import json # Not used after removing execute_tool_call and step
# import os # Not used
# import re # Not used
# import tempfile # Not used
# import textwrap # Not used
# import time # Not used
# from abc import ABC, abstractmethod # Not used
# from collections import deque # Not used
# from pathlib import Path # Not used
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional # Union, Tuple, Set, Generator, TypedDict - Not used
# import jinja2 # Not used
import yaml # Still used for loading prompt_templates
# from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder # Not used
# from jinja2 import StrictUndefined, Template # Not used
# from rich.rule import Rule # Not used
# from rich.text import Text # Not used
# from rich.panel import Panel # Not used
# from rich import box # Not used


if TYPE_CHECKING:
    import PIL.Image # Keep for type hinting if AsyncMultiStepAgent or its methods use it

# from src.memory import (ActionStep, ToolCall) # ToolCall is handled by AsyncMultiStepAgent, ActionStep not used
# from src.models import (ChatMessage) # Not used after removing step
from src.logger import (
    LogLevel, # Keep for __init__ type hint
)

from src.tools import Tool # Keep for __init__ type hint
from src.exception import (
    # AgentParsingError, # Not used after removing step
    AgentToolCallError, # Potentially kept if there are other methods, but likely unused. Removing for now.
    AgentToolExecutionError, # Potentially kept if there are other methods, but likely unused. Removing for now.
)
from .async_multistep_agent import AsyncMultiStepAgent, PromptTemplates # PromptTemplates needed for __init__ type hint
from src.models import Model # Keep for __init__ type hint, removed parse_json_if_needed
# from src.utils.agent_types import AgentAudio, AgentImage # Not used after removing step

from src.logger import logger # Keep logger, remove YELLOW_HEX

class ToolCallingAgent(AsyncMultiStepAgent): # Changed parent class
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: Optional[PromptTemplates] = None, # Made Optional explicit
        planning_interval: Optional[int] = None, # Made Optional explicit
        **kwargs: Any, # Added type hint for kwargs
    ):
        # Load prompt_templates from the specific YAML file for ToolCallingAgent
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("src.base.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )

        # Extract relevant parameters from kwargs for AsyncMultiStepAgent's __init__
        max_steps = kwargs.pop("max_steps", 20)
        add_base_tools = kwargs.pop("add_base_tools", False)
        # Ensure verbosity_level uses self.logger.level if available from a previous init step,
        # or a kwarg, or a default. Here, ToolCallingAgent doesn't init self.logger before super()
        # so we rely on kwargs or the default LogLevel.INFO.
        verbosity_level = kwargs.pop("verbosity_level", LogLevel.INFO)
        grammar = kwargs.pop("grammar", None)
        managed_agents = kwargs.pop("managed_agents", None) # Assuming this might be passed via kwargs
        step_callbacks = kwargs.pop("step_callbacks", None)

        # name and description can be passed via kwargs or default to None / class name
        name = kwargs.pop("name", self.__class__.__name__)
        description = kwargs.pop("description", None)


        super().__init__(
            tools=tools,
            model=model,
            config=None,
            prompt_templates=prompt_templates,
            max_steps=max_steps,
            add_base_tools=add_base_tools,
            verbosity_level=verbosity_level,
            grammar=grammar,
            managed_agents=managed_agents,
            step_callbacks=step_callbacks,
            planning_interval=planning_interval,
            name=name,
            description=description,
            **kwargs,
        )

    # initialize_system_prompt is removed
    # step method is removed (now inherited from AsyncMultiStepAgent)
    # _substitute_state_variables method is removed (now inherited from AsyncMultiStepAgent)
    # execute_tool_call method is removed (now inherited from AsyncMultiStepAgent)