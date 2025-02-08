import pytest
from unittest.mock import patch, MagicMock
from utils import build_prompt, parse_theta_response, call_llm, get_new_theta

def test_build_prompt():
    prompt = build_prompt(s=10.0, w=5.0, old_theta=0.5, step=1)
    assert isinstance(prompt, str)
    assert "Sheep (s): 10.00" in prompt
    assert "Wolves (w): 5.00" in prompt
    assert "Previous theta: 0.500" in prompt
    assert "Time step: 1" in prompt

def test_parse_theta_response():
    # Test valid responses
    assert parse_theta_response("0.5") == 0.5
    assert parse_theta_response("The theta should be 0.7") == 0.7
    assert parse_theta_response("1.5") == 1.0  # Should clamp to 1.0
    assert parse_theta_response("-0.5") == 0.0  # Should clamp to 0.0
    
    # Test invalid responses
    assert parse_theta_response("no number here", default=0.5) == 0.5
    assert parse_theta_response("", default=0.3) == 0.3

@pytest.mark.asyncio
@patch('openai.ChatCompletion.create')
async def test_call_llm(mock_create):
    # Mock the OpenAI response
    mock_response = MagicMock()
    mock_response["choices"] = [{"message": {"content": "0.5"}}]
    mock_create.return_value = mock_response

    response = call_llm("test prompt")
    assert response == "0.5"
    
    # Verify the API was called with correct parameters
    mock_create.assert_called_once()
    call_args = mock_create.call_args[1]
    assert "messages" in call_args
    assert call_args["temperature"] == 0.0

@pytest.mark.asyncio
@patch('utils.call_llm')
async def test_get_new_theta(mock_call_llm):
    mock_call_llm.return_value = "0.7"
    
    theta = get_new_theta(s=10.0, w=5.0, old_theta=0.5, step=1)
    assert 0.0 <= theta <= 1.0
    
    # Test with invalid LLM response
    mock_call_llm.return_value = "invalid response"
    theta = get_new_theta(s=10.0, w=5.0, old_theta=0.5, step=1)
    assert theta == 0.5  # Should return old_theta as default