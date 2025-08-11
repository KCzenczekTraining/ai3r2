"""
This script automates data processing and validation.
It downloads a text file, evaluates arithmetic questions,
updates the answers using OpenAI's GPT model, and submits a report to a Agents HQ.
The program uses Python 3.10 features and adheres to PEP8 standards for clean and maintainable code.

Steps:
1. Downloads a file from a specified URL.
2. Converts the file content into a Python dictionary.
3. Processes test data to create valid test data and questions for OpenAI.
4. Queries OpenAI's GPT model for answers to specific questions.
5. Sends a report to an external endpoint and logs the response.

Ensure the `.env` file contains the required environment variables:
- `MY_POLIGON_KEY`: Key for accessing the file.
- `AGENT_HQ`: Base URL for the agent.
- `OPENAI_API_KEY`: API key for OpenAI.
"""

import json
import logging
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from S01.utils_S01 import configure_logging


# Load environment variables
load_dotenv()


# Retrieve environment variables
MY_POLIGON_KEY = os.getenv("MY_POLIGON_KEY")
AGENT_HQ = os.getenv("AGENT_HQ")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPORT_ENDPOINT = os.getenv("REPORT_ENDPOINT")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure logging using the generic function
configure_logging("logs_s01e03.txt")


def download_file():
    """Download the file from the specified URL and save it to the E03 folder."""

    try:
        logging.info("Starting file download.")
        url = f"{AGENT_HQ}/data/{MY_POLIGON_KEY}/json.txt"
        response = requests.get(url)
        response.raise_for_status()

        file_path = "S01/E03/json.txt"
        with open(file_path, "w") as file:
            file.write(response.text)

        logging.info(f"File downloaded successfully and saved to {file_path}.")
        return file_path
    except requests.RequestException as exc:
        logging.error(f"Error downloading file: {exc}")
        raise


def parse_file_to_dict(file_path):
    """Convert the content of the downloaded file to a Python dictionary."""

    try:
        logging.info(f"Parsing file content from {file_path}.")
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.info("File content parsed successfully.")
        return data
    except json.JSONDecodeError as exc:
        logging.error(f"Error parsing file content: {exc}")
        raise


def process_test_data(data):
    """Process test data and create valid_data and question_to_llm lists."""

    logging.info("Processing test data.")
    test_data = data.get("test-data", [])
    question_to_llm = []

    for item in test_data:
        question = item.get("question")
        answer = item.get("answer")
        test = item.get("test")

        if question and answer:
            if answer != eval(question):
                logging.warning(
                    f"Answer mismatch for question. The question: {question} has a wrong answer: {answer}"
                )
                item["answer"] = eval(question)

        if test:
            question_to_llm.append(test.get("q"))

    logging.info("Test data processed successfully.")
    return test_data, question_to_llm


def query_openai(questions):
    """Query OpenAI's GPT model to answer the questions."""

    logging.info(f"Querying OpenAI with questions: {questions}")
    try:
        completion = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": "You are a helpful assistant. Keep the answer as short as possible. Answer as list of dictionaries.\n<answer in format>\n[{\"q\": question_1, \"a\": answer_1}, {\"q\": question_2, \"a\": answer_2}, ..., {\"q\": question_n, \"a\": answer_n}]\n</answer in format>\n"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": f"{questions}"
                }
            ]
            },
        ],
        text={
            "format": {
            "type": "text"
            }
        },
        temperature=1,
        max_output_tokens=100,
        )
        response = json.loads(completion.output_text)
        logging.info(f"Received response from OpenAI: {response}")
        return response
    except OpenAIError as exc:
        logging.error(f"Error querying OpenAI API: {exc}")
        return None


def update_answers(valid_data, answers):
    """Update 'a' in 'test' in 'valid_data' based on matching 'q' in 'answers' received from LLM.
    example of missing 'a':
    {
        "question": "53 + 44",
        "answer": 97,
        "test": {
            "q": "name of the 2020 USA president",
            "a": "???"
        }
    }
    """

    logging.info("Updating answers in 'test' dict in valid_data.")
    for item in valid_data:
        if "test" in item and "q" in item["test"]:
            test_question = item["test"]["q"]
            for answer in answers:
                if answer.get("q") == test_question:
                    item["test"]["a"] = answer.get("a")
                    logging.info(f"Updated test answer for question '{test_question}' with answer '{answer.get('a')}'.")
                    break


def send_report(valid_data):
    """Send report data to REPORT_ENDPOINT and display response containing 'FLG'."""

    logging.info("Sending report to REPORT_ENDPOINT.")
    try:
        report_endpoint = os.getenv("REPORT_ENDPOINT")
        payload = {
            "task": "JSON",
            "apikey": MY_POLIGON_KEY,
            "answer": {
                "apikey": MY_POLIGON_KEY,
                "description": "This is simple calibration data used for testing purposes. Do not use it in production environment!",
                "copyright": "Copyright (C) 2238 by BanAN Technologies Inc.",
                "test-data": valid_data
            }
        }
        response = requests.post(report_endpoint, json=payload)
        response.raise_for_status()
        response_data = response.json()

        # Look for phrase containing 'FLG'
        flg_phrase = next((value for key, value in response_data.items() if "FLG" in str(value)), None)
        if flg_phrase:
            logging.info(f"Found phrase with 'FLG': {flg_phrase}")
        else:
            logging.warning("No phrase with 'FLG' found in the response.")
    except requests.RequestException as exc:
        logging.error(f"Error sending report: {exc}")
        raise


def main():
    """Main function to orchestrate the process."""

    try:
        logging.info("Starting main execution.")
        
        file_path = download_file()

        data = parse_file_to_dict(file_path)

        valid_data, question_to_llm = process_test_data(data)

        answers = query_openai(question_to_llm)

        update_answers(valid_data, answers)
        logging.info("Data is verified and updated")

        send_report(valid_data)

        
    except Exception as exc:
        logging.error(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
