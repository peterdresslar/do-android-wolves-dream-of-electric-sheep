import json
from unittest.mock import MagicMock, patch

import pytest

from model.utils.utils import (
    WolfResponse,
    build_prompt_high_information,
    build_prompt_low_information,
    call_llm,
    call_llm_for_consent,
    get_wolf_response,
    parse_wolf_response,
)


def test_build_prompt_high_information():
    prompt = build_prompt_high_information(
        s=10.0, w=5.0, old_theta=0.5, step=1, sheep_max=20.0
    )
    assert isinstance(prompt, str)
    assert "Sheep (s): 10.00" in prompt
    assert "Wolves (w): 5.00" in prompt
    assert "Previous theta: 0.500" in prompt
    assert "Time step: 1" in prompt
    assert "Maximum sheep capacity (sheep_max): 20.00" in prompt


def test_build_prompt_low_information():
    prompt = build_prompt_low_information(
        s=10.0, w=5.0, delta_s=1.0, delta_w=0.5, old_aggression=0.5
    )
    assert isinstance(prompt, str)
    assert "Sheep: 10.00" in prompt
    assert "Wolves: 5.00" in prompt
    assert "Your previous theta: 0.50" in prompt


def test_parse_wolf_response():
    # Test valid JSON responses
    valid_response = '{"theta": 0.5, "explanation": "test", "vocalization": "howl"}'
    result = parse_wolf_response(valid_response)
    parsed = json.loads(valid_response)
    assert result.theta == parsed["theta"]
    assert result.explanation == parsed["explanation"]
    assert result.vocalization == parsed["vocalization"]

    # Test response clamping
    high_response = '{"theta": 1.5, "explanation": "too high"}'
    result = parse_wolf_response(high_response)
    assert result.theta == 1.0

    low_response = '{"theta": -0.5, "explanation": "too low"}'
    result = parse_wolf_response(low_response)
    assert result.theta == 0.0

    # Test invalid responses
    result = parse_wolf_response("invalid json", default=0.5)
    assert result.theta == 0.5


@patch("utils.openai.OpenAI")
def test_call_llm(mock_openai_class):
    # Create a mock client with the proper structure
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"theta": 0.5}'))]
    )
    mock_openai_class.return_value = mock_client

    response = call_llm("test prompt")
    assert response == '{"theta": 0.5}'
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch("utils.call_llm")
async def test_get_wolf_response(mock_call_llm):
    # Test verbose response
    mock_call_llm.return_value = (
        '{"theta": 0.7, "explanation": "test", "vocalization": "howl"}'
    )
    response = get_wolf_response(
        s=10.0, w=5.0, sheep_max=20.0, old_theta=0.5, step=1, respond_verbosely=True
    )
    assert isinstance(response, WolfResponse)
    assert 0.0 <= response.theta <= 1.0
    assert response.explanation == "test"
    assert response.vocalization == "howl"

    # Test non-verbose response
    mock_call_llm.return_value = '{"theta": 0.3}'
    response = get_wolf_response(
        s=10.0, w=5.0, sheep_max=20.0, old_theta=0.5, step=1, respond_verbosely=False
    )
    assert isinstance(response, WolfResponse)
    assert 0.0 <= response.theta <= 1.0
    assert response.explanation is None
    assert response.vocalization is None

    # Test with invalid LLM response
    mock_call_llm.return_value = "invalid response"
    response = get_wolf_response(s=10.0, w=5.0, sheep_max=20.0, old_theta=0.5, step=1)
    assert response.theta == 0.5  # Should return old_theta as default


@patch("utils.datetime")
@patch("utils.call_llm")
@patch("builtins.open")
def test_call_llm_for_consent(mock_open, mock_call_llm, mock_datetime):
    # Setup mocks
    mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"
    mock_call_llm.return_value = '{"consent": "yes", "date-time": "2024-01-01T00:00:00Z", "model-name": "test-model", "explanation": "test"}'
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    # Call the function
    call_llm_for_consent(model="test-model")

    # Assert the LLM was called with consent prompt
    mock_call_llm.assert_called_once()
    assert (
        "consent" in mock_call_llm.call_args[0][0]
    )  # Check if prompt contains "consent"

    # Assert file was written with correct name and content
    expected_filename = "docs/consent/consent-test-model-2024-01-01T00:00:00.json"
    mock_open.assert_called_once_with(expected_filename, "w")
    mock_file.write.assert_called_once_with(mock_call_llm.return_value)
