
import os
import logging
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI
from S01.utils_S01 import configure_logging


load_dotenv()
configure_logging("logs_langfuse_python.txt")


class OpenAIService:
    """
    Service for interacting with OpenAI API.
    """


    def __init__(self) -> None:
        """
        Initialize OpenAI client.
        """
        logging.info("Initializing OpenAIService and OpenAI client.")
        self.client: OpenAI = OpenAI()


    def completion(self, config: Dict[str, Any]) -> Any:
        """
        Generate a completion using OpenAI API.

        Args:
            config (Dict[str, Any]): Configuration for the completion.

        Returns:
            Any: Completion result from OpenAI API.
        """
        logging.info(f"Calling OpenAI API for completion with model: {config.get('model', 'gpt-4.1-nano')}")
        messages: Any = config['messages']
        model: str = config.get('model', 'gpt-4.1-nano')
        stream: bool = config.get('stream', False)
        json_mode: bool = config.get('jsonMode', False)

        logging.info(f"Generating completion with model={model}, stream={stream}, json_mode={json_mode}")

        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            response_format={'type': 'json_object'} if json_mode else {'type': 'text'}
        )
