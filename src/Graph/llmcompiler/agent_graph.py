from __future__ import annotations

from typing import Annotated, Iterable, Iterator, List, Optional, TypedDict, Dict, Any
import logging

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.Graph.llmcompiler.task_fetching_unit import plan_and_schedule
from src.Graph.llmcompiler.joiner import build_joiner
from src.logs.log import build_logger

class GraphState(TypedDict):
    # Reducer: any node that returns {"messages": [...]} gets appended automatically
    messages: Annotated[List[BaseMessage], add_messages]

class LLMCompilerAgent:
    """
    Encapsulates the LLMCompiler-style LangGraph:
      START -> plan_and_schedule -> joiner --(AIMessage)-> END
                                     \--(SystemMessage)-> plan_and_schedule (loop)
    """
    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        joiner_runnable=None,              # allow DI for tests/custom joiners
        plan_and_schedule_runnable=None,   # allow DI
    ) -> None:
        self.log = logger or build_logger(level=logging.INFO)
        self.joiner = joiner_runnable or build_joiner()
        self.plan_and_schedule = plan_and_schedule_runnable or plan_and_schedule
        self._chain = None  # compiled graph

    def compile(self) -> "LLMCompilerAgent":
        """Build and compile the stateful graph. Call once at startup."""
        self._chain = self._build_graph().compile()
        self.log.debug("[AGENT] Graph compiled.")
        return self

    def invoke(
            self,
            messages: List[BaseMessage],
            *,
            recursion_limit: int = 100,
            config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the graph once to completion; return final state."""
        self._ensure_compiled()
        self.log.debug("[AGENT.invoke] Start (messages=%d, recursion_limit=%d)", len(messages), recursion_limit)
        state = self._chain.invoke({"messages": messages}, {"recursion_limit": recursion_limit, **(config or {})})
        self.log.debug("[AGENT.invoke] Done. Final messages=%d", len(state.get("messages", [])))
        return state


    def stream(
        self,
        messages: List[BaseMessage],
        *,
        recursion_limit: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream node-by-node updates (useful for logs/UX)."""
        self._ensure_compiled()
        self.log.debug("[AGENT.stream] Start (messages=%d, recursion_limit=%d)", len(messages), recursion_limit)
        yield from self._chain.stream({"messages": messages}, {"recursion_limit": recursion_limit, **(config or {})})

    def graph_png(self) -> bytes:
        """Return a PNG of the compiled graph (for debugging/visualization)."""
        self._ensure_compiled()
        return self._chain.get_graph().draw_mermaid_png()

    # -------- helpers --------

    def _build_graph(self) -> StateGraph:
        self.log.debug("[AGENT._build_graph] Build nodes & edges")
        g = StateGraph(GraphState)

        # nodes
        g.add_node("plan_and_schedule", self.plan_and_schedule)  # planner + scheduler/executor
        g.add_node("joiner", self.joiner)

        # edges
        g.add_edge("plan_and_schedule", "joiner")

        def should_continue(state: GraphState):
            msgs = state["messages"]
            if msgs and isinstance(msgs[-1], AIMessage):
                # Last is AIMessage => joiner finalized the answer
                return END
            # Else: last is SystemMessage with feedback => replan
            return "plan_and_schedule"

        g.add_conditional_edges("joiner", should_continue)
        g.add_edge(START, "plan_and_schedule")
        return g

    def _ensure_compiled(self) -> None:
        if self._chain is None:
            raise RuntimeError("Agent graph is not compiled. Call .compile() first.")

# -------- convenience factory & helpers --------
def get_agent(logger: Optional[logging.Logger] = None) -> LLMCompilerAgent:
    """Singleton-ish factory to cache the compiled graph at process start."""
    agent = LLMCompilerAgent(logger=logger).compile()
    return agent


def make_human(text: str) -> HumanMessage:
    return HumanMessage(content=text)