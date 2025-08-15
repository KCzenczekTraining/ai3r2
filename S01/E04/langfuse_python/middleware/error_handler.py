from typing import Any
import logging

from flask import jsonify
from S01.utils_S01 import configure_logging

# Configure logging using the generic function
configure_logging("logs_langfuse_python.txt")

def error_handler(error: Any) -> Any:
    """
    Handle errors and return a JSON response.

    Args:
        error (Any): The error to handle.

    Returns:
        Any: JSON response with error details.
    """

    logging.info("Error handler invoked")  # Log when the error handler is called
    logging.error(f"Error handled: {error}")
    response = jsonify({'error': 'An error occurred while processing your request'})
    response.status_code = 500
    logging.info(f"Response prepared with status code {response.status_code}")  # Log the response status
    return response
