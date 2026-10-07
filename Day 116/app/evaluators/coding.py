"""
Automated coding evaluation pipeline: syntax checking and test case verification.
"""
import ast
import re
from typing import Dict, List, Any, Optional, Tuple


def extract_code(text: str) -> str:
    """
    Extracts executable Python code from raw model output:
    Looks for ```python ... ``` blocks, or lines starting from 'def ' or 'import '.
    """
    # Check for markdown fenced code blocks
    fenced_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Look for first 'def '
    lines = text.split("\n")
    code_lines = []
    capturing = False
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("import "):
            capturing = True
        if capturing:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return text.strip()


def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Python syntax using the abstract syntax tree (AST).
    Returns (is_valid, error_message).
    """
    if not code:
        return False, "Empty code snippet"
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_test_cases(
    code: str,
    test_cases: List[Dict[str, Any]],
    timeout_calls: int = 100000
) -> Dict[str, Any]:
    """
    Executes a generated function against input-output test cases in a restricted namespace.
    Restricts __builtins__ to safe mathematical and standard operations.
    """
    is_valid, syntax_err = validate_syntax(code)
    if not is_valid:
        return {
            "syntax_valid": False,
            "tests_passed": 0,
            "total_tests": len(test_cases),
            "pass_rate": 0.0,
            "error": syntax_err
        }

    # Safe builtins subset
    safe_builtins = {
        "range": range, "len": len, "sum": sum, "min": min, "max": max,
        "abs": abs, "all": all, "any": any, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "int": int, "float": float, "str": str, "bool": bool, "sorted": sorted,
        "reversed": reversed, "round": round
    }
    exec_globals = {"__builtins__": safe_builtins}

    try:
        exec(code, exec_globals)
    except Exception as e:
        return {
            "syntax_valid": True,
            "tests_passed": 0,
            "total_tests": len(test_cases),
            "pass_rate": 0.0,
            "error": f"Execution error during definition: {type(e).__name__}: {e}"
        }

    if not test_cases:
        # If no explicit test cases, valid syntax gets 100%
        return {
            "syntax_valid": True,
            "tests_passed": 1,
            "total_tests": 1,
            "pass_rate": 100.0,
            "error": None
        }

    passed = 0
    failure_reason = None

    for idx, tc in enumerate(test_cases):
        fn_name = tc.get("function_name")
        inputs = tc.get("inputs", [])
        expected = tc.get("expected")

        if fn_name not in exec_globals or not callable(exec_globals[fn_name]):
            return {
                "syntax_valid": True,
                "tests_passed": 0,
                "total_tests": len(test_cases),
                "pass_rate": 0.0,
                "error": f"Function '{fn_name}' not defined in output code."
            }

        func = exec_globals[fn_name]
        try:
            actual = func(*inputs)
            if actual == expected:
                passed += 1
            else:
                if failure_reason is None:
                    failure_reason = f"Test {idx+1} failed: expected {expected}, got {actual}"
        except Exception as e:
            if failure_reason is None:
                failure_reason = f"Test {idx+1} raised {type(e).__name__}: {e}"

    pass_rate = round((passed / max(1, len(test_cases))) * 100.0, 2)
    return {
        "syntax_valid": True,
        "tests_passed": passed,
        "total_tests": len(test_cases),
        "pass_rate": pass_rate,
        "error": failure_reason
    }


def evaluate_coding_example(
    prediction: str,
    test_cases: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates a single model prediction on a coding task.
    Combines code extraction, syntax validation, and test case execution.
    """
    extracted_code = extract_code(prediction)
    test_cases = test_cases or []
    result = execute_test_cases(extracted_code, test_cases)
    result["extracted_code"] = extracted_code

    # Calculate overall coding score (0 to 100)
    if not result["syntax_valid"]:
        score = 0.0
    elif not test_cases:
        score = 100.0  # Valid syntax without failing any constraints
    else:
        score = result["pass_rate"]

    result["score"] = score
    return result
