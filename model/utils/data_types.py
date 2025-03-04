"""
Shared data types used across the model.
This module contains dataclasses and other shared types to avoid circular imports.
"""

from dataclasses import dataclass

@dataclass
class WolfResponse:
    """Response from a wolf agent's decision process"""
    theta: float
    prompt: str | None = None
    explanation: str | None = None
    vocalization: str | None = None


@dataclass
class Usage:
    """Track token usage and cost for LLM calls"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Add token usage from a single LLM call"""
        from model.utils.llm_utils import calculate_cost
        
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost += calculate_cost(prompt_tokens, completion_tokens, model)
        self.calls += 1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "calls": self.calls,
        }

# Global usage tracking
current_usage = None

def set_current_usage(usage: Usage):
    """Set the global usage tracker"""
    global current_usage
    current_usage = usage

def get_current_usage() -> Usage:
    """Get the global usage tracker"""
    global current_usage
    return current_usage 