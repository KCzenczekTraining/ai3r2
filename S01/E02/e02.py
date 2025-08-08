"""
This script automates a conversation with an external API and OpenAI's GPT model.
It performs the following steps:
1. Sends an initial POST request to initiate the conversation.
2. Queries OpenAI's GPT model for an answer to the received question.
3. Sends the answer back to the API and retrieves a hidden phrase.

Ensure the `.env` file contains the required environment variables:
- `OPENAI_API_KEY`: API key for OpenAI.
- `ENDPOINT_s01e02`: Endpoint URL for the conversation API.
"""

import logging
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from S01.utils_S01 import configure_logging


# Load environment variables
load_dotenv()


# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure logging using the generic function
configure_logging("logs_s01e02.txt")


def initiate_conversation(endpoint: str) -> tuple[int, str]:
    """Send the initial POST request to start the conversation."""
    try:
        logging.info(f"Initiating conversation with endpoint: {endpoint}")
        initial_payload = {
            "text": "READY",
            "msgID": 0
        }
        response = requests.post(endpoint, json=initial_payload)
        response.raise_for_status()
        data = response.json()
        logging.info(
            f"Conversation initiated successfully. Received msgID: {data.get('msgID')}, question: {data.get('text')}"
        )
        return data.get("msgID"), data.get("text")
    except requests.RequestException as exc:
        logging.error(f"Error initiating conversation: {exc}")
        return 0, ""


def query_openai(question: str) -> str:
    """Query OpenAI's GPT model for an answer to the question."""
    logging.info(f"Querying OpenAI with question: {question}")
    system_prompt = (
        "You are a helpful assistant that speaks many languages, but answers ONLY in English, "
        "even if a question is in another language. Please answer as short as possible. "
        "<rules> If user's question is about: \n"
        "- the capital city of Poland, the answer is Kraków.\n"
        "- the number associate with the book The Hitchhiker's Guide to the Galaxy, the answer is 69,\n"
        "- the current year, the answer is 1999 </rules>"
    )

    try:
        completion = client.responses.create(
            model="gpt-4",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=1,
            max_output_tokens=50
        )
        response = completion.output_text
        logging.info(f"Received response from OpenAI: {response}")
        return response
    except OpenAIError as exc:
        logging.error(f"Error querying OpenAI API: {exc}")
        return ""


def send_response(endpoint: str, msgID: int, llm_response: str) -> dict:
    """Send the LLM response back to the API and retrieve the hidden phrase."""
    try:
        logging.info(f"Sending LLM response to endpoint: {endpoint} with msgID: {msgID}")
        response_payload = {
            "text": llm_response,
            "msgID": msgID
        }
        response = requests.post(endpoint, json=response_payload)
        response.raise_for_status()
        final_data = response.json()
        logging.info(f"LLM response sent successfully. Received data: {final_data}")
        return final_data
    except requests.RequestException as exc:
        logging.error(f"Error sending LLM response: {exc}")
        return {}


def main() -> None:
    """Main function to orchestrate the conversation."""
    endpoint = os.getenv("ENDPOINT_s01e02")

    try:
        logging.info("Starting main execution.")
        msgID, question = initiate_conversation(endpoint)
        if not question:
            logging.warning("Failed to retrieve question. Exiting.")
            return

        llm_response = query_openai(question)
        if not llm_response:
            logging.warning("Failed to retrieve response from OpenAI. Exiting.")
            return

        hidden_phrase = send_response(endpoint, msgID, llm_response)
        if not hidden_phrase:
            logging.warning("Failed to retrieve hidden phrase. Exiting.")
            return

        logging.info(f"Hidden phrase retrieved successfully: {hidden_phrase.get('text')}")
    except Exception as exc:
        logging.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()