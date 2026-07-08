"""Shared NanoChat conversation formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def normalize_messages(messages: Iterable[dict]) -> list[Message]:
    """Merge system messages into the next user message, matching SFT training."""
    processed: list[Message] = []
    system_parts: list[str] = []
    for raw in messages:
        role = str(raw.get("role", ""))
        content = str(raw.get("content", ""))
        if role == "system":
            system_parts.append(content)
            continue
        if system_parts and role == "user":
            content = "\n\n".join(system_parts) + "\n\n" + content
            system_parts = []
        elif system_parts:
            processed.append(Message("user", "\n\n".join(system_parts)))
            system_parts = []
        processed.append(Message(role, content))
    if system_parts:
        processed.append(Message("user", "\n\n".join(system_parts)))
    return processed


def validate_messages_for_generation(messages: list[Message]) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    for index, message in enumerate(messages):
        if message.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role at index {index}: {message.role!r}")
        expected = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise ValueError(
                f"message {index} has role {message.role!r}, expected {expected!r}"
            )
    if messages[-1].role != "user":
        raise ValueError("generation prompts must end with a user message")


def encode_chat_prompt(tokenizer, messages: Iterable[dict], token_budget: int) -> list[int]:
    """Encode messages and append assistant_start to prime generation.

    If the prompt exceeds the budget, earlier turns are dropped before the latest
    turn is truncated head+tail. This keeps the newest user request intact enough
    for smoke/regression evaluation while respecting the model context.
    """
    processed = normalize_messages(messages)
    validate_messages_for_generation(processed)

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    encoded_turns: list[list[int]] = []
    for message in processed:
        content = tokenizer.encode(message.content)
        if message.role == "user":
            encoded_turns.append([user_start, *content, user_end])
        else:
            encoded_turns.append([assistant_start, *content, assistant_end])

    fixed = [bos, assistant_start]
    if token_budget <= len(fixed):
        raise ValueError("prompt token budget is too small")

    selected: list[list[int]] = []
    used = len(fixed)
    for turn in reversed(encoded_turns):
        if used + len(turn) <= token_budget:
            selected.append(turn)
            used += len(turn)
        elif not selected:
            remaining = token_budget - used
            if remaining < 3:
                raise ValueError("prompt token budget cannot fit one user turn")
            start_token, end_token = turn[0], turn[-1]
            content_budget = remaining - 2
            content = turn[1:-1]
            head = int(content_budget * 0.7)
            truncated = content[:head] + content[-(content_budget - head):]
            selected.append([start_token, *truncated, end_token])
            used = token_budget
        else:
            break
    selected.reverse()
    if selected and selected[0][0] == assistant_start:
        selected = selected[1:]
    tokens = [bos]
    for turn in selected:
        tokens.extend(turn)
    tokens.append(assistant_start)
    return tokens
