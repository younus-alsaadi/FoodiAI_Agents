"""
The Planner is a runnable chain that:

Decides whether to plan (first time) or replan (if joiner gave feedback).

Builds the correct prompt with tool descriptions and context.

Calls the LLM (ChatGPT model).

Parses the LLM output into structured Tasks (with tool, args, dependencies).

"""

from typing import Sequence, List

from autogen.messages import BaseMessage
from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (FunctionMessage, SystemMessage, HumanMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool


from src.Graph.llmcompiler.output_parser import LLMCompilerPlanParser
from src.helpers.config import get_settings


from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from src.Graph.llmcompiler.tools.tool_test1 import get_math_tool
import os
import logging
from src.logs.log import build_logger
from src.Graph.llmcompiler.tools import get_nutrition_tool
from src.Graph.llmcompiler.tools.search_internet import TavilySearchTool
from src.Graph.llmcompiler.tools.search_youtube import YouTubeRecipeSearchTool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from src.Graph.llmcompiler.plan_pretty import format_plan



log = build_logger(level=logging.DEBUG)
app_settings=get_settings()

def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    """Create a planner that can plan out a series of steps to accomplish a goal."""

    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n"
        for i, tool in enumerate(
            tools
        )  # +1 to offset the 0 starting index, we want it count normally from 1.
    )

    log.debug("[Planner] tool_descriptions call")
    log.debug(tool_descriptions)
    log.debug("=" * 20)

    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
                  + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )

    log.debug("[Planner] planner_prompt call")
    log.debug(planner_prompt)
    log.debug("=" * 20)

    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
               "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
               'You MUST use these information to create the next plan under "Current Plan".\n'
               ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
               " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
               " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    log.debug("[Planner] replanner_prompt call")
    log.debug(replanner_prompt)
    log.debug("=" * 20)

    # So the Planner always starts with should_replan.
    def should_replan(state: list):
        # Context is passed as a system message
        sh= isinstance(state[-1], SystemMessage)
        log.debug(f"[Planner] should_replan call: {sh}")
        log.debug("=" * 20)
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        log.debug("[Planner] wrap_messages call")
        log.debug("=" * 20)
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        log.debug("[Planner] wrap_and_get_last_index call and last msg")
        log.debug(f"[Planner]messages: {state}")
        log.debug("=" * 20)
        return {"messages": state}

    log.debug("RunnableBranch call\n")
    log.debug( RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools))
    log.debug("=" * 20)
    log.debug("[Planner] create_planner call end\n")
    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )



def build_planner():
    """
    Factory that returns a fully-wired planner runnable.
    Use this from other modules instead of duplicating setup code.
    """
    # env (so downstream libs can read keys)

    log.debug("#" * 20)
    log.debug("[Planner] planner builld start call")
    log.debug("#"*20)

    os.environ['OPENAI_API_KEY'] = app_settings.OPENAI_API_KEY
    os.environ['TAVILY_API_KEY'] = app_settings.TAVILY_API_KEY
    os.environ['YOUTUBE_API_KEY'] = app_settings.YOUTUBE_API_KEY

    os.environ["LANGCHAIN_TRACING_V2"] = app_settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = app_settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = app_settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = app_settings.LANGCHAIN_PROJECT


    # tools
    calculate_nutrition = get_nutrition_tool(ChatOpenAI(model="gpt-4o"))
    search_tool = TavilySearchTool(max_results=3).tavily_as_tool()
    youtube_tool = YouTubeRecipeSearchTool(max_results=3).youtube_as_tool()



    log.debug("[Planner] Math tool call")
    log.debug("="*20)

    log.debug("[Planner] TavilySearch tool call")
    log.debug("=" * 20)

    tools = [search_tool, calculate_nutrition,youtube_tool]

    # prompt + llm
    prompt = hub.pull("wfh/llm-compiler")

    log.debug("[Planner] prompt wfh/llm-compiler call")
    log.debug(prompt)
    log.debug("=" * 20)

    llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")

    log.debug("[Planner] ChatOpenAI call")
    log.debug("=" * 20)

    # return runnable planner
    return create_planner(llm, tools, prompt)


def render_planner_prompt(
    base_prompt: ChatPromptTemplate,
    tools: Sequence[BaseTool],
    state: List[BaseMessage],
    replan: bool = False,
):
    tool_descriptions = "\n".join(
        f"{i+1}. {t.description}\n" for i, t in enumerate(tools)
    )
    prompt = base_prompt.partial(
        replan="" if not replan else (
            ' - You are given "Previous Plan" ... Do not repeat task indices.'
        ),
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    # IMPORTANT: actually render with the current messages/state
    rendered: List[BaseMessage] = prompt.format_messages(messages=state)

    print("="*20, "RENDERED PROMPT", "="*20)
    for i, msg in enumerate(rendered):
        role = type(msg).__name__.replace("Message","").lower()
        print(f"\n[{i}] {role.upper()}:\n{msg.content}")
    print("="*56)



if __name__ == "__main__":
    os.environ['OPENAI_API_KEY']=app_settings.OPENAI_API_KEY
    os.environ['TAVILY_API_KEY']=app_settings.TAVILY_API_KEY
    os.environ['YOUTUBE_API_KEY'] = app_settings.YOUTUBE_API_KEY

    os.environ["LANGCHAIN_TRACING_V2"] = app_settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = app_settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = app_settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = app_settings.LANGCHAIN_PROJECT



    calculate_nutrition = get_nutrition_tool(ChatOpenAI(model="gpt-4o"))
    search_inInternet_tool = TavilySearchTool(max_results=3).tavily_as_tool()
    search_youtube_tool = YouTubeRecipeSearchTool(max_results=3).youtube_as_tool()



    tools = [search_inInternet_tool, calculate_nutrition,search_youtube_tool]

    prompt = hub.pull("wfh/llm-compiler")

    llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")
    # This is the primary "agent" in our application
    planner = create_planner(llm, tools, prompt)

    # print(planner)

    state = [HumanMessage(content="How to make biryani rice?")]
    #render_planner_prompt(base_prompt=prompt, tools=tools, state=state, replan=False)
    task=planner.invoke(state)

    print(format_plan(task))

    # example_question = "How to make berani rice?"
    #
    # for task in planner.stream([HumanMessage(content=example_question)]):
    #     print(task["tool"], task["args"])
    #     print("---"*20)







