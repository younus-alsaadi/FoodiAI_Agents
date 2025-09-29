"""
So now we have the planning and initial execution done. We need a component to process these outputs and either:
Respond with the correct answer.
Loop with a new plan.

After the planner + executor produce tool outputs, the Joiner:
Looks at the most recent slice of the conversation (last Human + any newer tool messages).
Asks an LLM (with structured output) to decide:
 -Finish → return the final answer, or
 -Replan → give feedback so the Planner can continue.
Converts that decision into messages:
 -Finish → AIMessage(final answer)
 -Replan → SystemMessage("Context from last attempt: <feedback>")
(It also adds an AIMessage("Thought: …") describing the reasoning.)
"""
from typing import Union, List
from autogen.messages import BaseMessage
from langchain import hub
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.Graph.llmcompiler.task_fetching_unit import plan_and_schedule
import logging
from src.logs.log import build_logger



log = build_logger(level=logging.DEBUG)


class FinalResponse(BaseModel):
    """The final response/answer."""
    response: str
    def model_post_init(self, __context):
        log.debug(f"[JOINER.FinalResponse] response preview: {self.response[:80]!r}")

class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )
    def model_post_init(self, __context):
        log.debug(f"[JOINER.Replan] feedback: {self.feedback!r}")


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


# Always include an AIMessage with the Joiner’s thought.
def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    log.debug(f"[JOINER.parse] Decision: action={type(decision.action).__name__} thought={decision.thought!r}")
    if isinstance(decision.action, Replan):
        log.debug(f"[JOINER.parse(Replan)] Feedback: {decision.action.feedback!r}")
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        }
    else:
        log.debug(f"[JOINER.parse] Final response len={len(decision.action.response)}")
        return {"messages": response + [AIMessage(content=decision.action.response)]}

# select only the most recent messages
def select_recent_messages(state) -> dict:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:   # loop backwards
        selected.append(msg)
        if isinstance(msg, HumanMessage):  # stop when we reach the last user input
            break
    log.debug(f"[JOINER.select_recent_messages] Selected {len(selected)} msgs (from last Human onward)")
    return {"messages": selected[::-1]}




def build_joiner():
    """
    Factory that builds and returns the joiner runnable.
    This component decides whether to replan or finalize the answer.
    """
    log.debug("*$#"*20)
    log.debug("[JOINER] Building joiner...")
    log.debug("*$#" * 20)

    joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
    log.debug("[JOINER] joiner_prompt call...")
    log.debug(joiner_prompt)
    log.debug("=" * 60)


    llm = ChatOpenAI(model="gpt-4o")
    log.debug("[joiner] ChatOpenAI call for joiner ...")
    log.debug("=" * 60)

    runnable = joiner_prompt | llm.with_structured_output(
        JoinOutputs, method="function_calling"
    )

    joiner = select_recent_messages | runnable | _parse_joiner_output

    log.debug("[JOINER] full joiner call done ...")
    log.debug(joiner)
    log.debug("=" * 60)

    return joiner


# if __name__ == "__main__":
#     joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
#
#
#     llm = ChatOpenAI(model="gpt-4o")
#
#     runnable = joiner_prompt | llm.with_structured_output(
#         JoinOutputs, method="function_calling"
#     )
#
#     joiner = select_recent_messages | runnable | _parse_joiner_output
#     print(runnable)

#     example_question = "What's the temperature in Hamburg in c raised to the 2rd power?"
#     tool_messages = plan_and_schedule.invoke(
#         {"messages": [HumanMessage(content=example_question)]}
#     )["messages"]
#
#     input_messages = [HumanMessage(content=example_question)] + tool_messages
#
#     joiner = build_joiner()
#     result = joiner.invoke({"messages": input_messages})
#
#     print(result)