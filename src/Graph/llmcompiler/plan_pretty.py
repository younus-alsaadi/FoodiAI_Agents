# plan_pretty.py
from __future__ import annotations
from typing import Any, Dict, List

def _tool_name(t: Any) -> str:
    try:
        # LangChain StructuredTool
        return getattr(t, "name", str(t))
    except Exception:
        return str(t)

def _fmt_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            if v is None:
                continue
            parts.append(f"{k}={repr(v)}")
        return ", ".join(parts)
    if isinstance(args, (list, tuple)):
        return ", ".join(repr(x) for x in args)
    return repr(args)

def format_plan(tasks: List[Dict[str, Any]], markdown: bool = False) -> str:
    """Render LLMCompiler tasks into a clean text or Markdown list."""
    lines: List[str] = []
    for t in tasks:
        idx = t.get("idx", "?")
        tool = t.get("tool")
        name = _tool_name(tool)
        args = _fmt_args(t.get("args"))
        deps = t.get("dependencies") or []
        dep_str = f"  ⟶ deps: {','.join(map(str, deps))}" if deps else ""
        line = f"{idx}. {name}({args}){dep_str}"
        lines.append(line)

    if markdown:
        return "\n".join(f"- {ln}" for ln in lines)
    return "\n".join(lines)
