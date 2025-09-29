"""
This component schedules the tasks. It receives a stream of tools of the following format:
{
    tool: BaseTool,
    dependencies: number[],
}
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from math import lgamma
from typing import Any, Dict, Iterable, List, Union, TypedDict
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from src.Graph.llmcompiler.output_parser import Task
from langchain_core.runnables import chain as as_runnable
import itertools

from src.Graph.llmcompiler.planner import build_planner
import logging
from src.logs.log import build_logger



log = build_logger(level=logging.DEBUG)

# Build the planner once for this process
planner = build_planner()


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:

    """
    If your state had FunctionMessage(..., idx=1, content="42") and FunctionMessage(..., idx=2, content="21"),
    this returns {1: "42", 2: "21"}.
    """
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content

    log.debug(f"[TFU] Current observations from past messages: {results}")
    log.debug("=" * 20)

    return results

class SchedulerInput(TypedDict):
    messages: List[BaseMessage]  # the running conversation/state
    tasks: Iterable[Task]        # tasks to run now (from the plan/DAG)


def _execute_task(task, observations, config):
    """

    Resolve args first (so "${1}" becomes "42" etc.).

    Then call the tool’s .invoke(...).

    On any failure, return a string that starts with ERROR(...) (so the Joiner can decide what to do).

    """

    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use  # (edge case: not a real tool object)

    args = task["args"]



    log.debug("=" * 20)
    try:
        # Replace ${k} with the actual output from observations
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {k: _resolve_arg(v, observations) for k, v in args.items()}
        else:
            resolved_args = args  # as-is

        log.debug(f"[TFU] Running task idx={task['idx']} tool={tool_to_use.name} raw_args={args}")
        log.debug(f"[TFU] Resolved args={resolved_args}")
    except Exception as e:
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. " \
               f"Args could not be resolved. Error: {repr(e)}"

    try:

        return tool_to_use.invoke(resolved_args, config)  # run the tool!

    except Exception as e:
        return f"ERROR(Failed to call {tool_to_use.name} with args {args}. " \
               f"Args resolved to {resolved_args}. Error: {repr(e)})"




def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):

    """
    If arg="Price is ${2} USD", and observations[2]="199", this returns "Price is 199 USD".

    If no observation exists yet, it leaves the placeholder unchanged (which will likely cause an error later—by design).

    """

    ID_PATTERN = r"\$\{?(\d+)\}?"  # matches $1 or ${1}

    def replace_match(m):
        idx = int(m.group(1))                      # the number inside
        return str(observations.get(idx, m.group(0)))  # replace if we have it

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)  # replace in strings
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):

    """
    Wrapped as a Runnable: you call schedule_task.invoke({"task": t, "observations": obs}).

    After running, it writes the result into observations[task_idx].
    """

    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback
        observation = traceback.format_exception()
    observations[task["idx"]] = observation  # save the output by task id



# Helper for tasks that must wait for deps
def schedule_pending_task(task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
    """
    Runs in a background thread.

    Loops until all dependencies are present in observations, then executes the task.
    """

    while True:
        deps = task["dependencies"]
        if deps and any([dep not in observations for dep in deps]):
            time.sleep(retry_after)  # not ready → wait, then check again
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        log.debug(f"[TFU] [WAIT] Task idx={task['idx']} waiting for deps {deps}, current obs keys={list(observations.keys())}")
        log.debug("=" * 20)
        break



@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]

    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        log.debug(f"[TFU][SCHEDULER] Received {tasks}")
        log.debug("=" * 20)
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            log.debug(task["tool"].name)
            log.debug("=" * 20)
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke, dict(task=task, observations=observations)))
            log.debug(f"[TFU] [SCHEDULER] Task {task['idx']} ({task_names[task['idx']]}) deps={deps}")
            log.debug("=" * 20)

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)


    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    log.debug(f"[TFU][SCHEDULER] All tasks completed, new observations={new_observations}")
    log.debug("=" * 20)
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages

# streams a plan from the Planner and immediately hands tasks to the scheduler

@as_runnable
def plan_and_schedule(state):

    log.debug("#"*20)
    log.debug("[TFU] plan_and_schedule call from task_fetching_unit")
    log.debug("#" * 20)


    messages = state["messages"]
    log.debug(f"[TFU] Starting planning + scheduling for messages: {messages}")
    log.debug("=" * 20)

    tasks = planner.stream(messages)
    log.debug("[TFU] Current tasks call:")
    for task in tasks:
        log.debug(f"[TFU] task : {task}")
        log.debug("=" * 20)



    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )

    log.debug(f"[TFU]Final scheduled tool messages: {scheduled_tasks}")
    log.debug("="*20)

    return {"messages": scheduled_tasks}

# if __name__ == "__main__":
#     example_question = "What's the temperature in Hamburg raised to the 2rd power?"
#
#     tool_messages = plan_and_schedule.invoke(
#         {"messages": [HumanMessage(content=example_question)]}
#     )["messages"]






