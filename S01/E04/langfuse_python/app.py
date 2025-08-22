"""
Flask API for orchestrating chat completions and observability using Langfuse and OpenAI.

Services:
- ChatService: Handles chat completions using OpenAI models.
- LangfuseService: Manages traces, spans, and generations for observability and debugging.

Main Steps:
1. Receives chat requests via POST /api/chat.
2. Creates a trace and span for each request using Langfuse.
3. Calls OpenAI to generate a chat completion.
4. Updates and finalizes spans and traces with input/output data.
5. Returns the completion, conversation ID, and trace ID in the response.
6. Handles graceful shutdown and error logging.

Logging:
- All major steps and errors are logged for traceability and debugging.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import signal
import uuid
import logging
from typing import Dict, Any, List
from flask import Flask, request, jsonify

from chat_service import ChatService
from langfuse_service import LangfuseService
from middleware.error_handler import error_handler
from S01.utils_S01 import configure_logging

configure_logging("logs_langfuse_python.txt")


app: Flask = Flask(__name__)
app.register_error_handler(Exception, error_handler)

chat_service: ChatService = ChatService()
langfuse_service: LangfuseService = LangfuseService()


@app.route('/api/chat', methods=['POST'])
def chat() -> Any:
    """
    Handle chat requests and generate responses using Langfuse and OpenAI.

    Returns:
        Any: JSON response containing the completion and trace ID.
    """
    logging.info(f"Received new chat request.")
    data: Dict[str, Any] = request.get_json()
    user_messages: List[Dict[str, Any]] = data.get('messages', [])
    conversation_id: str = data.get('conversation_id', str(uuid.uuid4()))

    logging.info(f"Creating trace for chat request.")
    trace_id: str = langfuse_service.create_trace({
        'id': str(uuid.uuid4()),
        'name': 'Chaty-Chat',
        'session_id': conversation_id,
        'user_id': data.get('user_id', 'unknown')
    })

    try:
        system_messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': 'You are a helpful assistant.', 'name': 'Norika'}
        ]

        all_messages: List[Any] = [*system_messages, *user_messages]

        generated_response: List[str] = []

        logging.info("Creating main span for chat completion.")
        main_span = langfuse_service.create_span('Main Completion', all_messages)


        try:
            logging.info("Calling OpenAI for chat completion.")
            main_completion = chat_service.completion(all_messages, 'gpt-4.1-nano')
            langfuse_service.finalize_span(all_messages, main_completion)
            main_message: str = main_completion.choices[0].message.content
            all_messages.append(main_message)
            generated_response.append(main_message)
            main_span.update(output=main_message)
        except Exception as exc:
            main_span.update(error=str(exc))
            logging.error(f"Error during OpenAI completion: {exc}")
            raise


        logging.info("Finalizing trace for chat request.")
        langfuse_service.finalize_trace(user_messages, generated_response)


        return jsonify({
            'completion': main_completion.choices[0].message.content,
            'conversation_id': conversation_id,
            'trace_id': trace_id
        })

    except Exception as exc:
        logging.error(f"Error in chat endpoint: {exc}")
        return jsonify({'error': str(exc)}), 500


def graceful_shutdown(*args: Any) -> None:
    """
    Handle graceful shutdown of the application.
    """
    logging.info("Graceful shutdown initiated.")
    langfuse_service.shutdown()
    exit(0)


signal.signal(signal.SIGINT, graceful_shutdown)


if __name__ == '__main__':
    app.run(port=3004)
