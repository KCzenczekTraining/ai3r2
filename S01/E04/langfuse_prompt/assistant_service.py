
"""
AssistantService module: Handles prompt-based chat completions using OpenAI and Langfuse for observability.

Key logic points are logged for debugging and traceability.
"""

import logging
from typing import Any, Dict, List
from openai_service import OpenAIService
from langfuse_service import LangfuseService
from langfuse import get_client
from S01.utils_S01 import configure_logging


configure_logging("logs_langfuse_prompts.txt")


class AssistantService:
    """
    Service for generating chat completions using OpenAI and tracking with Langfuse.
    """


    def __init__(self, openai_service: OpenAIService, langfuse_service: LangfuseService) -> None:
        """
        Initialize AssistantService with OpenAI and Langfuse services.
        """
        self.openai_service: OpenAIService = openai_service
        self.langfuse_service: LangfuseService = langfuse_service
        self.langfuse_client = get_client()
        logging.info("AssistantService initialized.")


    def answer(self, messages: List[Dict[str, Any]], session_id: str, user_id: str) -> Any:
        """
        Generate a chat completion using a prompt from Langfuse and OpenAI.

        Args:
            messages (List[Dict[str, Any]]): List of user messages.
            session_id (str): Session identifier.
            user_id (str): User identifier.

        Returns:
            Any: OpenAI completion object.
        """
        logging.info(f"Generating answer for session_id={session_id}, user_id={user_id}")
        prompt = self.langfuse_client.get_prompt('my_prompt_from_LF')
        system_message: Any = prompt.compile()[0]
        thread: List[Dict[str, Any]] = [system_message] + [msg for msg in messages if msg.get('role') != 'system']
        try:
            logging.info("Creating Langfuse span for generation.")
            with self.langfuse_service.create_span({'name': 'Span for Generation'}) as span_g:
                span_g.update_trace(
                    session_id=session_id,
                    user_id=user_id
                )
                logging.info("Creating Langfuse generation and calling OpenAI.")
                with self.langfuse_service.create_generation(
                    name='here_i_call_llm',
                    model='gpt-4.1-nano',
                    input={'messages': thread},
                ) as gen:
                    completion: Any = self.openai_service.completion({'messages': thread})
                    gen.update(
                        output=completion.choices[0].message.content,
                        usage_details={
                            'promptTokens': completion.usage.prompt_tokens,
                            'completionTokens': completion.usage.completion_tokens,
                            'totalTokens': completion.usage.total_tokens
                        }
                    )
                logging.info("Answer generated successfully.")
                return completion
        except Exception as error:
            logging.error(f"Error in AssistantService answer: {error}")
            raise
