"""
Unit tests for coding evaluators (AST, sandbox execution) and instruction constraint verifiers.
"""
import pytest
from app.evaluators.coding import (
    extract_code,
    validate_syntax,
    execute_test_cases,
    evaluate_coding_example
)
from app.evaluators.instruction import (
    validate_json_output,
    count_bullet_points,
    evaluate_instruction_following
)


def test_extract_code_from_markdown_fences():
    text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nHope this helps!"
    extracted = extract_code(text)
    assert extracted == "def add(a, b):\n    return a + b"


def test_extract_code_from_def_statement():
    text = "Sure! Here is the implementation:\ndef multiply(x, y):\n    return x * y"
    extracted = extract_code(text)
    assert extracted.startswith("def multiply(x, y):")


def test_validate_syntax_valid_code():
    code = "def square(x):\n    return x ** 2"
    is_valid, err = validate_syntax(code)
    assert is_valid is True
    assert err is None


def test_validate_syntax_invalid_code():
    code = "def broken(x\n    return x +"
    is_valid, err = validate_syntax(code)
    assert is_valid is False
    assert "SyntaxError" in err


def test_validate_syntax_empty_code():
    is_valid, err = validate_syntax("")
    assert is_valid is False


def test_execute_test_cases_success():
    code = "def add(a, b):\n    return a + b"
    test_cases = [
        {"function_name": "add", "inputs": [2, 3], "expected": 5},
        {"function_name": "add", "inputs": [-1, 1], "expected": 0}
    ]
    res = execute_test_cases(code, test_cases)
    assert res["syntax_valid"] is True
    assert res["pass_rate"] == 100.0
    assert res["tests_passed"] == 2


def test_execute_test_cases_failure():
    code = "def add(a, b):\n    return a - b"  # Bug
    test_cases = [{"function_name": "add", "inputs": [2, 3], "expected": 5}]
    res = execute_test_cases(code, test_cases)
    assert res["tests_passed"] == 0
    assert res["pass_rate"] == 0.0


def test_execute_test_cases_syntax_error():
    code = "def incomplete("
    test_cases = [{"function_name": "incomplete", "inputs": [1], "expected": 1}]
    res = execute_test_cases(code, test_cases)
    assert res["syntax_valid"] is False
    assert "SyntaxError" in res["error"]


def test_evaluate_coding_example_full_pass():
    prediction = "```python\ndef double(n):\n    return n * 2\n```"
    test_cases = [
        {"function_name": "double", "inputs": [5], "expected": 10},
        {"function_name": "double", "inputs": [0], "expected": 0}
    ]
    res = evaluate_coding_example(prediction, test_cases=test_cases)
    assert res["syntax_valid"] is True
    assert res["tests_passed"] == 2
    assert res["score"] == 100.0


def test_validate_json_output_valid_dict():
    text = '{"name": "MiniGPT", "version": "1.0"}'
    is_valid, err = validate_json_output(text)
    assert is_valid is True
    assert err is None


def test_validate_json_output_valid_list():
    text = '[1, 2, 3, "apple"]'
    is_valid, err = validate_json_output(text)
    assert is_valid is True


def test_validate_json_output_with_surrounding_text():
    text = 'Here is your JSON response:\n```json\n{"status": "ok", "code": 200}\n```\nDone.'
    is_valid, err = validate_json_output(text)
    assert is_valid is True


def test_validate_json_output_invalid_json():
    text = '{"status": "incomplete", '
    is_valid, err = validate_json_output(text)
    assert is_valid is False
    assert "JSON parsing failed" in err


def test_validate_json_output_required_keys_present():
    text = '{"model": "MiniGPT", "loss": 0.42}'
    is_valid, err = validate_json_output(text, required_keys=["model", "loss"])
    assert is_valid is True


def test_validate_json_output_required_keys_missing():
    text = '{"model": "MiniGPT"}'
    is_valid, err = validate_json_output(text, required_keys=["model", "loss"])
    assert is_valid is False
    assert "Missing required JSON keys" in err


def test_count_bullet_points_hyphen_and_asterisk():
    text = "- Point one\n* Point two\n- Point three"
    assert count_bullet_points(text) == 3


def test_count_bullet_points_numbered_list():
    text = "1. First step\n2. Second step\n3. Third step\n4. Fourth step"
    assert count_bullet_points(text) == 4


def test_evaluate_instruction_following_json_rule():
    prompt = "Return your response strictly in JSON format."
    pred = '{"result": "success"}'
    res = evaluate_instruction_following(prompt, pred)
    assert res["is_compliant"] is True
    assert res["score"] == 100.0


def test_evaluate_instruction_following_word_count_cap():
    prompt = "Answer in exactly four words."
    pred_ok = "Python is very versatile"
    res_ok = evaluate_instruction_following(prompt, pred_ok)
    assert res_ok["is_compliant"] is True

    pred_long = "Python is a very versatile and widely adopted programming language in data science."
    res_long = evaluate_instruction_following(prompt, pred_long)
    assert res_long["is_compliant"] is False
    assert res_long["score"] < 100.0


def test_evaluate_instruction_following_capital_letters():
    prompt = "Answer in CAPITAL LETTERS only."
    pred_ok = "ARTIFICIAL INTELLIGENCE"
    res_ok = evaluate_instruction_following(prompt, pred_ok)
    assert res_ok["is_compliant"] is True

    pred_bad = "Artificial Intelligence"
    res_bad = evaluate_instruction_following(prompt, pred_bad)
    assert res_bad["is_compliant"] is False
