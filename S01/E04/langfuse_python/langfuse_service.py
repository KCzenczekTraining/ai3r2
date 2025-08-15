import os
import json
from typing import List, Dict, Any
import logging

from dotenv import load_dotenv
from langfuse import Langfuse
from S01.utils_S01 import configure_logging

load_dotenv()

# Configure logging using the generic function
configure_logging("logs_langfuse_python.txt")


class LangfuseService:
    """
    Service for interacting with Langfuse SDK.
    Handles traces, spans, and generations for observability.
    """


    def __init__(self):
        """
        Initialize Langfuse client with environment variables.
        """
        logging.info("Initializing LangfuseService and Langfuse client.")
        self.langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST')
        )
        if os.getenv('NODE_ENV') == 'development':
            self.langfuse.debug()


    def create_trace(self, options: Dict[str, str]) -> str:
        """
        Create a new trace and return its ID.

        Args:
            options (Dict[str, str]): Trace options including name, session_id, and user_id.

        Returns:
            str: Trace ID.
        """
        logging.info(f"Creating new trace with options: {options}")
        with self.langfuse.start_as_current_span(name=options["name"]) as span:
            span.update_trace(session_id=options["session_id"], user_id=options.get("user_id"))
        logging.info(f"Trace created with ID: {span.trace_id}")
        return span.trace_id


    def create_span(self, name: str, input: Any = None):
        """
        Create a new span within the current trace.

        Args:
            name (str): Name of the span.
            input (Any, optional): Input data for the span.

        Returns:
            Span: The created span object.
        """
        logging.info(f"Creating new span: {name}")
        with self.langfuse.start_as_current_span(name=name) as span:
            if input is not None:
                span.update(input=json.dumps(input))
            logging.info(f"Span created: {span}")
            return span


    def finalize_span(self, all_messages: List[Dict[str, Any]], output: Any) -> None:
        """
        Finalize a span for an LLM call.

        Args:
            all_messages (List[Dict[str, Any]]): Input messages for the LLM.
            output (Any): Output from the LLM.
        """
        logging.info("Finalizing span for LLM call.")
        with self.langfuse.start_as_current_generation(
            name='llm-call',
            model=output.model,
            input={"messages": all_messages}
        ) as generation:
            generation.update(
                output=output.choices[0].message.content,
                usage_details={"input_tokens": 10, "output_tokens": 25}  # Example token usage
            )
        logging.info("Span finalized")


    def finalize_trace(self, original_messages: List[Dict[str, Any]], generated_responce: List[Dict[str, Any]]) -> None:
        """
        Finalize the trace with input and output messages.

        Args:
            original_messages (List[Dict[str, Any]]): Original input messages.
            generated_responce (List[Dict[str, Any]]): Generated output messages.
        """
        logging.info("Finalizing trace with input and output messages.")
        input_messages = [msg for msg in original_messages if msg.get('role') != 'system']
        with self.langfuse.start_as_current_span(name="finalize-trace") as span:
            span.update_trace(
                input=json.dumps(input_messages),
                output=json.dumps(generated_responce)
            )
        self.langfuse.flush()
        logging.info("Trace finalized")


    def shutdown(self) -> None:
        """
        Shutdown the Langfuse client.
        """
        logging.info("Shutting down Langfuse client.")
        self.langfuse.shutdown()
        logging.info("Langfuse client shut down")
