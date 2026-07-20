"""Post-generation answer validation and cleanup."""


def validate_answer(answer: str, sources: list, *, max_new_tokens: int = 512) -> str:
    """Post-generation answer validation and cleanup.

    - Strips whitespace and model artifacts
    - Returns refusal message if answer is empty or near-empty
    - Truncates overly long answers to max_new_tokens * 4 chars
    """
    if not answer:
        return "I couldn't generate a valid answer. Please try rephrasing your question."

    # Strip model artifacts and excessive whitespace
    answer = answer.strip()
    for artifact in ["<|end|>", "</s>", "[END]", "[/INST]"]:
        answer = answer.replace(artifact, "")
    answer = answer.strip()

    # Near-empty after cleanup
    if len(answer) < 10:
        return "I couldn't generate a meaningful answer. Please try rephrasing your question."

    # Truncate overly long answers (safety cap)
    max_chars = max_new_tokens * 4
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0] + "..."

    return answer
