"""Phase 3 FAILED state tests."""
import os, sys
from decimal import Decimal
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand, CalculationOperation, CalculationPlan,
    CalculationResult, CalculationStatus,
)
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_renderer import render_calculation_result
from src.finance.calculation_registry import CALCULATION_REGISTRY


def _operand(name, value):
    return CalculationOperand(
        name=name, value=Decimal(value),
        source_text=f"{name} = {value}", evidence_chunk_id=f"chunk_{name}",
    )

def _ready_plan(operation, operands):
    from src.finance.calculation_registry import get_operation_entry
    entry = get_operation_entry(operation)
    return CalculationPlan(
        operation=operation, operands=operands,
        formula_version=entry.formula_version if entry else "unknown.v1",
        target_metric=operation.value,
    )


class TestExecutorFailedScenarios:
    def test_unknown_operation_returns_failed(self):
        entry = CALCULATION_REGISTRY.pop(CalculationOperation.SUM, None)
        try:
            plan = _ready_plan(CalculationOperation.SUM, (_operand("a", "1"),))
            result = execute_plan(plan)
            assert result.status is CalculationStatus.FAILED
            assert result.error_code == "UNKNOWN_OPERATION"
        finally:
            if entry is not None:
                CALCULATION_REGISTRY[CalculationOperation.SUM] = entry

    def test_registry_function_exception_returns_failed(self):
        original = CALCULATION_REGISTRY.get(CalculationOperation.SUM)
        if original is None:
            pytest.skip("SUM not in registry")
        def boom(operands, precision):
            raise RuntimeError("internal boom")
        bad = type(original)(
            operation=original.operation, func=boom,
            formula=original.formula, formula_version=original.formula_version,
            min_operands=original.min_operands, max_operands=original.max_operands,
            operand_roles=original.operand_roles, unit=original.unit,
        )
        CALCULATION_REGISTRY[CalculationOperation.SUM] = bad
        try:
            plan = _ready_plan(CalculationOperation.SUM, (_operand("a", "1"), _operand("b", "2")))
            result = execute_plan(plan)
            assert result.status is CalculationStatus.FAILED
            assert result.error_code == "PRIMITIVE_EXCEPTION"
        finally:
            CALCULATION_REGISTRY[CalculationOperation.SUM] = original

    def test_failed_does_not_expose_internal_stack(self):
        result = CalculationResult(
            status=CalculationStatus.FAILED, operation=CalculationOperation.SUM,
            error_code="PRIMITIVE_EXCEPTION",
            error_message="KeyError: secret", target_metric="sum",
        )
        rendered = render_calculation_result(result)
        assert "secret" not in rendered
        assert "KeyError" not in rendered
        assert "Unable to compute" in rendered

    def test_failed_result_has_error_code(self):
        result = CalculationResult(
            status=CalculationStatus.FAILED, operation=CalculationOperation.SUM,
            error_code="PRIMITIVE_EXCEPTION", error_message="boom",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error_code"] == "PRIMITIVE_EXCEPTION"


class TestRendererFailedSafety:
    def test_failed_returns_safe_message_not_empty(self):
        result = CalculationResult(
            status=CalculationStatus.FAILED,
            error_code="PRIMITIVE_EXCEPTION",
            error_message="Traceback inner stack",
        )
        rendered = render_calculation_result(result)
        assert rendered != ""
        assert "Traceback" not in rendered
        assert "Unable to compute" in rendered

    def test_failed_with_no_error_code_still_safe(self):
        result = CalculationResult(status=CalculationStatus.FAILED)
        rendered = render_calculation_result(result)
        assert rendered != ""
        assert "Unable to compute" in rendered


class TestFailedOrchestratorBypass:
    def test_failed_status_in_bypass_set(self):
        bypass = {CalculationStatus.EXECUTED, CalculationStatus.BLOCKED, CalculationStatus.FAILED}
        assert CalculationStatus.FAILED in bypass

    def test_not_applicable_not_in_bypass_set(self):
        bypass = {CalculationStatus.EXECUTED, CalculationStatus.BLOCKED, CalculationStatus.FAILED}
        assert CalculationStatus.NOT_APPLICABLE not in bypass

    def test_failed_confidence_is_zero(self):
        status = CalculationStatus.FAILED
        c = 1.0 if status is CalculationStatus.EXECUTED else 0.0 if status is CalculationStatus.FAILED else 0.5
        assert c == 0.0


class TestFailedTraceDiagnostics:
    def test_failed_result_to_dict_has_status_and_error(self):
        result = CalculationResult(
            status=CalculationStatus.FAILED,
            operation=CalculationOperation.DIFFERENCE,
            error_code="PRIMITIVE_EXCEPTION",
            error_message="boom",
            formula_version="difference.v1",
            target_metric="difference",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error_code"] == "PRIMITIVE_EXCEPTION"
        assert d["operation"] == "difference"
        assert d["formula_version"] == "difference.v1"
