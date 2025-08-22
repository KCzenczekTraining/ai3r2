
"""
Langfuse Prompt Application

This application provides a Flask-based API for chat completions with robust observability and tracing using Langfuse and OpenAI. 
It is designed for experimentation, debugging, and analysis of conversational AI flows, with a focus on transparency and traceability.

Application Structure:
- app.py: Main entry point. Defines the Flask app and the /api/chat endpoint, orchestrating the request flow.
- assistant_service.py: Contains the AssistantService class, which manages the logic for generating responses using OpenAI and integrates with Langfuse for observability.
- openai_service.py: Handles direct communication with the OpenAI API, encapsulating prompt and completion logic.
- langfuse_service.py: Manages trace and span creation, finalization, and logging using the Langfuse SDK.
- utils_S01.py: Provides utility functions, including logging configuration.

Step-by-Step Request Flow:
1. The client sends a POST request to /api/chat with chat messages and optional session/user information.
2. The Flask app receives the request and extracts the relevant data (messages, session_id, user_id).
3. A new trace is created in Langfuse to record the start of the chat interaction.
4. The AssistantService is called to generate a response, which internally uses OpenAIService for completions and LangfuseService for observability.
5. The trace is finalized in Langfuse, recording both the input messages and the generated response.
6. The API returns the generated response and session ID to the client as JSON.
7. All key steps are logged for debugging and observability.
"""

import logging
import uuid

from flask import Flask, jsonify, request
from typing import Any, Dict, List

from assistant_service import AssistantService
from langfuse_service import LangfuseService
from openai_service import OpenAIService
from S01.utils_S01 import configure_logging


configure_logging("logs_langfuse_prompts.txt")


app: Flask = Flask(__name__)
langfuse_service: LangfuseService = LangfuseService()
openai_service: OpenAIService = OpenAIService()
assistant_service: AssistantService = AssistantService(openai_service, langfuse_service)


@app.route('/api/chat', methods=['POST'])
def chat() -> Any:
    """
    Handle chat requests and generate responses using Langfuse and OpenAI.

    Returns:
        Any: JSON response containing the generated response and session ID.
    """
    logging.info("Received new chat request.")
    data: Dict[str, Any] = request.get_json()
    session_id: str = data.get('session_id', str(uuid.uuid4()))
    user_id: str = data.get('user_id', 'test_user')

    data_messages: List[Dict[str, Any]] = data.get('messages', [])
    messages: List[Dict[str, Any]] = [msg for msg in data_messages if msg.get('role') != 'system']

    try:
        logging.info("Creating trace for chat request.")
        langfuse_service.create_trace(
            {
                'name': 'Main trace creation',
                'input': (messages[-1]['content'][:45] if messages else ''),
                'session_id': session_id,
                'user_id': user_id
            }
        )

        logging.info("Calling AssistantService.answer().")
        answer: Any = assistant_service.answer(messages, session_id, user_id)

        logging.info("Finalizing trace for chat request.")
        langfuse_service.finalize_trace(
            {
                'name': 'Main trace finalization',
                'session_id': session_id,
                'user_id': user_id,
                'original_messages': messages,
                'generated_response': answer.choices[0].message.content
            }
        )

        logging.info("Returning chat response.")
        return jsonify({'response': answer.choices[0].message.content, 'session_id': session_id})
    except Exception as error:
        logging.error(f'Error in chat processing: {error}')
        return jsonify({'error': 'An error occurred while processing your request'}), 500


if __name__ == '__main__':
    app.run(port=3000)
