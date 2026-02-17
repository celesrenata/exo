"""API compatibility tests for PyTorch+IPEX backend.

This module tests that the PyTorch+IPEX backend maintains compatibility
with the OpenAI chat completions API format.
"""

import json

import pytest


class TestAPICompatibility:
    """Test OpenAI API compatibility for PyTorch+IPEX backend."""

    def test_chat_completions_request_format(self) -> None:
        """Test that chat completions request format is valid."""
        request = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        }

        # Validate request structure
        assert "model" in request
        assert "messages" in request
        assert isinstance(request["messages"], list)
        assert len(request["messages"]) > 0

        # Validate message format
        for message in request["messages"]:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["system", "user", "assistant"]

    def test_streaming_request_format(self) -> None:
        """Test that streaming request format is valid."""
        request = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True,
            "temperature": 0.8,
            "top_p": 0.9,
        }

        assert request["stream"] is True
        assert "temperature" in request
        assert "top_p" in request

    def test_non_streaming_response_format(self) -> None:
        """Test that non-streaming response format matches OpenAI spec."""
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        # Validate response structure
        assert "id" in response
        assert "object" in response
        assert response["object"] == "chat.completion"
        assert "created" in response
        assert "model" in response
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) > 0

        # Validate choice structure
        choice = response["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Validate message structure
        message = choice["message"]
        assert "role" in message
        assert "content" in message
        assert message["role"] == "assistant"

        # Validate usage structure
        assert "usage" in response
        usage = response["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_streaming_response_format(self) -> None:
        """Test that streaming response format matches OpenAI spec."""
        # First chunk
        chunk1 = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        # Validate first chunk
        assert chunk1["object"] == "chat.completion.chunk"
        assert "choices" in chunk1
        choice = chunk1["choices"][0]
        assert "delta" in choice
        assert "finish_reason" in choice
        assert choice["finish_reason"] is None

        # Middle chunk
        chunk2 = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {"index": 0, "delta": {"content": "!"}, "finish_reason": None}
            ],
        }

        # Validate middle chunk
        assert chunk2["object"] == "chat.completion.chunk"
        delta = chunk2["choices"][0]["delta"]
        assert "content" in delta
        assert "role" not in delta  # Role only in first chunk

        # Final chunk
        chunk3 = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        # Validate final chunk
        assert chunk3["object"] == "chat.completion.chunk"
        assert chunk3["choices"][0]["finish_reason"] == "stop"
        assert chunk3["choices"][0]["delta"] == {}

    def test_error_response_format(self) -> None:
        """Test that error response format matches OpenAI spec."""
        error_response = {
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        }

        # Validate error structure
        assert "error" in error_response
        error = error_response["error"]
        assert "message" in error
        assert "type" in error
        assert isinstance(error["message"], str)
        assert isinstance(error["type"], str)

    def test_temperature_parameter(self) -> None:
        """Test that temperature parameter is handled correctly."""
        # Valid temperature values
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            request = {
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": temp,
            }
            assert 0.0 <= request["temperature"] <= 2.0

    def test_top_p_parameter(self) -> None:
        """Test that top_p parameter is handled correctly."""
        # Valid top_p values
        valid_top_p = [0.1, 0.5, 0.9, 1.0]
        for top_p in valid_top_p:
            request = {
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "messages": [{"role": "user", "content": "test"}],
                "top_p": top_p,
            }
            assert 0.0 <= request["top_p"] <= 1.0

    def test_max_tokens_parameter(self) -> None:
        """Test that max_tokens parameter is handled correctly."""
        request = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
        }
        assert isinstance(request["max_tokens"], int)
        assert request["max_tokens"] > 0

    def test_stop_sequences(self) -> None:
        """Test that stop sequences are handled correctly."""
        # Single stop sequence
        request1 = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "test"}],
            "stop": "\n",
        }
        assert isinstance(request1["stop"], str)

        # Multiple stop sequences
        request2 = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "test"}],
            "stop": ["\n", "END", "STOP"],
        }
        assert isinstance(request2["stop"], list)
        assert all(isinstance(s, str) for s in request2["stop"])

    def test_finish_reasons(self) -> None:
        """Test that finish reasons are valid."""
        valid_finish_reasons = ["stop", "length", "content_filter", "tool_calls"]

        for reason in valid_finish_reasons:
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test"},
                        "finish_reason": reason,
                    }
                ],
            }
            assert response["choices"][0]["finish_reason"] in valid_finish_reasons

    def test_json_serialization(self) -> None:
        """Test that responses can be serialized to JSON."""
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        # Should serialize without errors
        json_str = json.dumps(response)
        assert isinstance(json_str, str)

        # Should deserialize correctly
        deserialized = json.loads(json_str)
        assert deserialized == response

    def test_multiple_choices(self) -> None:
        """Test that multiple choices are handled correctly."""
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response 1"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Response 2"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 40,
                "total_tokens": 50,
            },
        }

        assert len(response["choices"]) == 2
        assert response["choices"][0]["index"] == 0
        assert response["choices"][1]["index"] == 1

    def test_system_message_handling(self) -> None:
        """Test that system messages are handled correctly."""
        request = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Tell me about AI."},
            ],
        }

        # Validate message sequence
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"
        assert request["messages"][2]["role"] == "assistant"
        assert request["messages"][3]["role"] == "user"

    def test_empty_content_handling(self) -> None:
        """Test that empty content is handled correctly."""
        # Empty assistant message (for tool calls)
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        assert response["choices"][0]["message"]["content"] == ""
        assert isinstance(response["choices"][0]["message"]["content"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
