"""Phase 4 grounded-answer validation package.

Public modules:
- ``validation_policy``    : intent-aware ValidationPolicy configuration.
- ``answerability``        : pre-generation AnswerabilityEvaluator.
- ``claim_extractor``      : deterministic ExtractedClaim extraction.
- ``numeric_claim_validator``: numeric value grounding checks.
- ``unit_period_validator`` : unit / scale / currency / period checks.
- ``citation_validator``    : source object and claim-support checks.
- ``calculation_validator`` : CalculationResult consistency checks.
- ``unsupported_claim_validator``: high-confidence unsupported-claim flags.
- ``response_validator``    : aggregates all validators into a verdict.
- ``response_repair``       : single deterministic repair (no LLM).
- ``validation_pipeline``   : GroundedValidationPipeline facade.

Dependency direction: ``domain <- validation <- application <- services/api``.
This package must NOT import from ``src.services``, ``src.application``,
or ``src.api``.
"""
