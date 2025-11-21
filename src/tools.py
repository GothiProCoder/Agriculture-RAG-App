# src/tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.logger import logger

# --- Schema for Python Tool ---
class PythonInput(BaseModel):
    code: str = Field(description="Valid Pandas Python code. NOT a list. A single executable string.")

@tool(args_schema=PythonInput)
def python_analyst_tool(code: str, df_context=None) -> str:
    """
    Executes pandas code. The dataframe is available as 'df'.
    """
    logger.info(f"🐍 Executing Python Code: {code}")
    
    # Sanitize: Handle cases where LLM sends markdown blocks
    clean_code = code.strip()
    if clean_code.startswith("```python"):
        clean_code = clean_code.split("```python")[1].split("```")[0].strip()
    elif clean_code.startswith("```"):
        clean_code = clean_code.split("```")[1].split("```")[0].strip()
        
    # CRITICAL FIX: Ensure input is string
    if not isinstance(clean_code, str):
        return "ERROR: Input code must be a string, not a list or object."

    local_vars = {"df": df_context, "result": None}
    
    try:
        exec(clean_code, {}, local_vars)
        result = local_vars.get("result")
        if result is None:
            return "Error: The code ran but variable 'result' was None. Did you assign 'result = ...'?"
        return str(result)
    except Exception as e:
        logger.error(f"❌ Code Execution Failed: {e}")
        # We return the EXACT error so the LLM can self-correct
        return f"PYTHON_ERROR: {type(e).__name__}: {str(e)}"