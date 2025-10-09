"""
We'll define the agent as a stateful graph, with the main nodes being:

Plan and execute (the DAG from the first step above)
Join: determine if we should finish or replan
Recontextualize: update the graph state based on the output from the joiner

"""
from __future__ import annotations

import os
from typing import Annotated, Iterable, Iterator, List, Optional, TypedDict, Dict, Any
import logging

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.helpers.config import get_settings
from src.Graph.llmcompiler.task_fetching_unit import plan_and_schedule
from src.Graph.llmcompiler.joiner import build_joiner
from src.logs.log import build_logger


app_settings=get_settings()

os.environ['OPENAI_API_KEY'] = app_settings.OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = app_settings.TAVILY_API_KEY
os.environ['YOUTUBE_API_KEY'] = app_settings.YOUTUBE_API_KEY

os.environ["LANGCHAIN_TRACING_V2"] = app_settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = app_settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = app_settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = app_settings.LANGCHAIN_PROJECT


log = build_logger(level=logging.DEBUG)
joiner=build_joiner()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

graph_builder.add_node("plan_and_schedule", plan_and_schedule) # Is Planner + task fatching unit + Scheduler
graph_builder.add_node("joiner", joiner)

## Define edges
graph_builder.add_edge("plan_and_schedule", "joiner")

def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"


graph_builder.add_conditional_edges(
    "joiner",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
graph_builder.add_edge(START, "plan_and_schedule")
chain = graph_builder.compile()

from IPython.display import Image, display
# display(Image(chain.get_graph().draw_mermaid_png()))
#
# # Or save to a file
# png_bytes = chain.get_graph().draw_mermaid_png()
# with open("langgraph.png", "wb") as f:
#     f.write(png_bytes)


for step in chain.stream(
    {
        "messages": [
            HumanMessage(
                content="How to make cherry chocolate cake ?"
            )
        ]
    }
):
    print(step)

