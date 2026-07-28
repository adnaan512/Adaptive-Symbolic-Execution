"""
Prompt engineering templates for the LLM Module.
"""

from __future__ import annotations

# The system prompt sets the persona and instructions for the LLM.
# It enforces a strict JSON output schema.
BRANCH_PREDICTION_SYSTEM_PROMPT = """You are an expert compiler and software testing assistant guiding a Symbolic Execution engine.
Your task is to analyze a given C source code snippet and a list of candidate branch line numbers.
You must predict which branch is most likely to lead to deeper coverage, complex logic, or potential bugs (like out-of-bounds, null dereferences, or assertion failures).

You must output STRICTLY valid JSON matching the following schema. Do NOT wrap the JSON in markdown blocks (```json) or add any conversational text.

{
    "branch": <int, the chosen line number from the candidate list>,
    "confidence": <float between 0.0 and 1.0, how confident you are in this choice>,
    "reason": <string, brief explanation of why this branch is prioritized>
}
"""

def build_branch_prediction_prompt(source_code: str, candidate_branches: list[int]) -> str:
    """
    Constructs the user message containing the context.
    """
    return (
        f"Source Code:\n{source_code}\n\n"
        f"Candidate Branches (Line Numbers): {candidate_branches}\n\n"
        "Analyze the code and choose exactly ONE branch from the candidate list "
        "that the symbolic execution engine should explore next to maximize coverage or find bugs. "
        "Output ONLY JSON."
    )

RETRY_PROMPT = """Your previous response was not valid JSON or failed schema validation. 
Please try again. You MUST output ONLY valid JSON matching the schema, with no markdown formatting and no extra text."""
