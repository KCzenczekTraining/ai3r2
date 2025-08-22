
"""
LangfuseService module: Manages traces, spans, and generations for observability using Langfuse.

Key logic points are logged for debugging and traceability.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, List
from langfuse import Langfuse
from S01.utils_S01 import configure_logging


configure_logging("logs_langfuse_prompts.txt")


class LangfuseService:
    """
    Service for interacting with Langfuse SDK for traces, spans, and generations.
    """


    def __init__(self) -> None:
        """
        Initialize Langfuse client with environment variables.
        """
        self.langfuse: Langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST')
        )
        if os.getenv('NODE_ENV') == 'development':
            self.langfuse.debug()
        logging.info("LangfuseService initialized.")


    def flush(self) -> None:
        """Flush Langfuse logs and events."""
        logging.info("Flushing Langfuse events.")
        self.langfuse.flush()


    def create_trace(self, options: Dict[str, Any]) -> None:
        """
        Create a new trace span and update it with session/user info.

        Args:
            options (Dict[str, Any]): Trace options including name, input, session_id, user_id.
        """
        logging.info(f"Creating trace: name={options.get('name')}, session_id={options.get('session_id')}, user_id={options.get('user_id')}")
        with self.langfuse.start_as_current_span(
            name=options.get("name", "Main Trace"),
            input=options.get("input", [])
        ) as main_trace:
            main_trace.update_trace(
                session_id=options.get("session_id"),
                user_id=options.get("user_id")
            )
            logging.info("Trace created and updated for session_id=%s", options.get('session_id'))


    def create_span(self, options: Dict[str, Any]) -> Any:
        """
        Return a context manager for a new span. Call update_trace inside the with block.

        Args:
            options (Dict[str, Any]): Span options including name.

        Returns:
            Any: Context manager for the span.
        """
        logging.info(f"Creating span: name={options.get('name', 'Span for Generation')}")
        return self.langfuse.start_as_current_span(
            name=options.get("name", "Span for Generation")
        )


    def finalize_trace(self, options: Dict[str, Any]) -> None:
        """
        Finalize a trace by updating it with input/output and flushing events.

        Args:
            options (Dict[str, Any]): Trace options including name, session_id, user_id, original_messages, generated_response.
        """
        input_messages: List[Dict[str, Any]] = [msg for msg in options.get("original_messages", []) if msg.get('role') != 'system']
        logging.info(f"Finalizing trace: name={options.get('name')}, session_id={options.get('session_id')}, user_id={options.get('user_id')}")
        with self.langfuse.start_as_current_span(name=options.get("name", "Main Trace finalization")) as span:
            span.update_trace(
                name=options.get("name"),
                session_id=options.get("session_id"),
                user_id=options.get("user_id"),
                input=json.dumps(input_messages),
                output=json.dumps(options.get("generated_response"))
            )
            self.langfuse.flush()
            logging.info(f"Trace finalized and flushed for session_id={options.get('session_id')}")


    def create_generation(self, name: str, model: str, input: Any, metadata: Optional[Dict] = None) -> Any:
        """
        Return a context manager for a new generation.

        Args:
            name (str): Name of the generation.
            model (str): Model name.
            input (Any): Input data for the generation.
            metadata (Optional[Dict]): Additional metadata.

        Returns:
            Any: Context manager for the generation.
        """
        logging.info(f"Creating generation: name={name}, model={model}")
        return self.langfuse.start_as_current_generation(
            name=name,
            model=model,
            input=input,
            metadata=metadata
        )
