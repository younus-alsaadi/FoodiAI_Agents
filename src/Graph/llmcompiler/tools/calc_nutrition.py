import math
import os
import re
from typing import List, Optional, Dict

import numexpr
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.helpers.config import get_settings


class NutritionInfo(BaseModel):
    calories: float = Field(..., description="Total energy in kcal")
    protein: float = Field(..., description="Protein in grams")
    fat: float = Field(..., description="Fat in grams")
    carbs: float = Field(..., description="Carbohydrates in grams")

class NutritionEstimate(BaseModel):
    reasoning: str
    code: str

_NUTRITION_DESCRIPTION = (
    "nutrition(recipe: str, context: Optional[list[str]]) -> dict:\n"
    " - Estimates calories, protein, fat, carbs for a given recipe.\n"
    " - `recipe` must be a plain-text ingredient list (e.g. '2 eggs, 100g rice').\n"
    " - You can optionally provide extra context with known nutrition facts (e.g. '1 egg=70 kcal').\n"
    " - Returns a structured dictionary with numeric values."
)

_SYSTEM_PROMPT = """Translate a meal recipe into TOTAL nutrition for the whole batch (not per serving) unless explicitly stated.
Assume standard items unless specified. Make assumptions explicit (e.g., "rice=COOKED 100 g", "1 tbsp oil=~13.5 g").

Rules:
- Give a single numeric payload: ExecuteCode({{code: "calories=<kcal>, protein=<g>, fat=<g>, carbs=<g>"}}).
- Enforce 4-4-9 consistency: calories ≈ 4*protein + 9*fat + 4*carbs within ±5%.
- List the key assumptions in reasoning (raw vs cooked, density conversions).
- Convert spoons/cups to grams using common culinary averages.
- If the recipe is ambiguous, choose the MOST common interpretation and state it.

Format:
Question: <recipe text in the user message>
OptionalContext: <provided via an extra system message>

Output:
ExecuteCode({{code: "calories=<number>, protein=<g>, fat=<g>, carbs=<g>"}})
Answer: Total = <kcal> (Protein: <g>, Fat: <g>, Carbs: <g>)

Example:
Question: 2 eggs, 100 g rice (COOKED), 1 tbsp olive oil
ExecuteCode({{code: "calories=389, protein=15g, fat=23g, carbs=29g"}})
Answer: Total = 389 kcal (Protein: 15 g, Fat: 23 g, Carbs: 29 g)
"""



_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions. 
Use it to substitute into any ${{#}} variables or other words in the recipe. 

${context}

Note: context variables are not predefined in the nutrition code. 
You must extract the relevant ingredient amounts (grams, ml, pieces, etc.) and directly put them in the structured nutrition output 
(calories, protein, fat, carbs)."""


_NUM_RE = r"([-+]?[0-9]*\.?[0-9]+)"


def _parse_kv_payload(code: str) -> Dict[str, float]:
    text = code.strip().lower()
    def grab(key: str) -> float:
        m = re.search(rf"{key}\s*=\s*{_NUM_RE}", text)
        if not m:
            # allow 'g' suffix for macros
            m = re.search(rf"{key}\s*=\s*{_NUM_RE}\s*g\b", text)
        if not m:
            raise ValueError(f"Missing or invalid '{key}' in code: {code!r}")
        return max(0.0, float(m.group(1)))
    return {
        "calories": grab("calories"),
        "protein": grab("protein"),
        "fat": grab("fat"),
        "carbs": grab("carbs"),
    }

def _reconcile_calories(vals, tol=0.05):
    p, f, c = vals["protein"], vals["fat"], vals["carbs"]
    kcal_from_macros = 4*p + 9*f + 4*c
    if vals["calories"] <= 0 or abs(vals["calories"] - kcal_from_macros) > tol * max(1.0, kcal_from_macros):
        vals["calories"] = round(kcal_from_macros, 1)
    return vals


def get_nutrition_tool(llm: ChatOpenAI) -> StructuredTool:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{recipe}"),
            MessagesPlaceholder(variable_name="context_msgs", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output(NutritionEstimate, method="function_calling")

    def nutrition(
        recipe: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, float]:
        chain_input = {"recipe": recipe}
        if context:
            ctx = "\n".join(x for x in context if x and x.strip()).strip()
            if ctx:
                chain_input["context_msgs"] = [
                    SystemMessage(content=_ADDITIONAL_CONTEXT_PROMPT.format(context=ctx))
                ]
        est: NutritionEstimate = extractor.invoke(chain_input, config)
        try:
            vals = _parse_kv_payload(est.code)
            vals=_reconcile_calories(vals)
        except Exception as e:
            # Surface parse error cleanly
            return {"error": f"Failed to parse nutrition code: {e}"}
        return vals

    return StructuredTool.from_function(
        name="nutrition",
        func=nutrition,
        description=_NUTRITION_DESCRIPTION,
    )

if __name__ == "__main__":
    app_settings = get_settings()
    os.environ['OPENAI_API_KEY'] = app_settings.OPENAI_API_KEY

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Step 2: build the nutrition tool
    nutrition_tool = get_nutrition_tool(llm)

    # Step 3: call the tool with a recipe
    result = nutrition_tool.invoke(
        {
            "recipe": "2 eggs, 100 g rice, 1 tbsp olive oil",
            "context": ["1 egg = 70 kcal, 6g protein, 5g fat"],
        }
    )

    print("Nutrition result:", result)