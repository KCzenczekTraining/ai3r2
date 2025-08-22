
import json
import logging
from typing import Any, Dict
from S01.E04.langfuse_python.openai_service import OpenAIService
from S01.utils_S01 import configure_logging


configure_logging("logs_langfuse_python.txt")


class ChatService:
    """
    Service for handling chat completions using OpenAI.
    """


    def __init__(self) -> None:
        """
        Initialize OpenAI service.
        """
        logging.info("Initializing ChatService and OpenAIService.")
        self.openai_service: OpenAIService = OpenAIService()


    def completion(self, messages: Any, model: str) -> Any:
        """
        Generate a completion using OpenAI.

        Args:
            messages (Any): Input messages for the completion.
            model (str): Model to use for the completion.

        Returns:
            Any: Completion result from OpenAI.
        """
        logging.info(f"Generating completion for model: {model}")
        if isinstance(messages, dict):
            messages = json.dumps(messages)

        result: Any = self.openai_service.completion({
            'messages': messages,
            'model': model,
            'stream': False,
            'jsonMode': False
        })
        logging.info(f"Completion generated successfully for model: {model}")
        return result
