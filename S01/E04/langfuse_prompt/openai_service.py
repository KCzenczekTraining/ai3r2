
"""
OpenAIService module: Handles OpenAI completions and (optionally) embeddings, with logging and error handling.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import logging
from typing import Any, Dict
from openai import OpenAI
from S01.utils_S01 import configure_logging


configure_logging("logs_langfuse_prompts.txt")


class OpenAIService:
    """
    Service for interacting with OpenAI API for completions.
    """


    def __init__(self) -> None:
        """Initialize OpenAI client."""
        self.client: OpenAI = OpenAI()
        logging.info("OpenAIService initialized.")


    def completion(self, config: Dict[str, Any]) -> Any:
        """
        Generate a completion using OpenAI API.

        Args:
            config (Dict[str, Any]): Configuration for the completion.

        Returns:
            Any: Completion result from OpenAI API.
        """
        messages: Any = config.get('messages', [])
        model: str = config.get('model', 'gpt-4.1-nano')
        stream: bool = config.get('stream', False)
        temperature: int = config.get('temperature', 0)
        max_tokens: int = config.get('maxTokens', 4096)
        json_mode: bool = config.get('jsonMode', False)
        response_format: Dict[str, str] = (
            {'type': 'json_object'} if json_mode else {'type': 'text'}
        )
        logging.info(
            f"Calling OpenAI completion: model={model}, stream={stream}, temperature={temperature}, "
            f"max_tokens={max_tokens}, json_mode={json_mode}"
        )
        try:
            completion: Any = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            logging.info(f"OpenAI completion successful for model={model}")
            return completion
        except Exception as error:
            logging.error(f"Error in OpenAI completion: {error}")
            raise
