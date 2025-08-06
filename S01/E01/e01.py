"""
This script automates the login process to a system that requires answering a dynamic question.
It uses the OpenAI API to generate an answer based on the question fetched from the login page. 
No additional websearch is required as LLM is able to answer the question directly as questions 
based on historical events.

Key Features:
1. Fetches a dynamic question from the login page using BeautifulSoup.
2. Queries OpenAI's GPT model to generate an answer to the question.
3. Submits login credentials and the generated answer to the system.
4. Processes the response to extract and display secret data, including content enclosed in double curly braces (e.g., {{content}}).
5. Opens the secret page in the default web browser for further inspection.

Ensure the `.env` file contains the required environment variables:
- `OPENAI_API_KEY`: API key for OpenAI.
- `USERNAME_s01e01`: Username for login.
- `PASSWORD_s01e01`: Password for login.
- `LOGIN_URL_s01e01`: URL of the login page.
"""


import os
import requests
import sys
import webbrowser
import tempfile
import re

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


# Load environment variables
load_dotenv()


# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def fetch_question(login_url: str) -> str | None:
    """Fetch the dynamic question from the website using BeautifulSoup."""
    try:
        response = requests.get(login_url, timeout=10)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")

            # Find the question section using id='human-question'
            question_element = soup.find(id='human-question')
            if question_element:
                question_text = question_element.get_text(strip=True)
                question_only = question_text[9:]
                return question_only
        else:
            print(f"Error: Received status code {response.status_code} from {login_url}")
    except requests.RequestException as e:
        print(f"Error: Failed to fetch the question due to {e}")

    return None


def get_answer_from_llm(question: str) -> str | None:
    """Get the answer to the question from OpenAI's GPT model."""
    try:
        response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "input_text",
                    "text": "You are a historian. Answer with only the year."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                    }
                ]
            }
        ],
        text={
            "format": {
            "type": "text"
            }
        },
        temperature=1,
        max_output_tokens=50,
        store=True
        )
        return response.output_text
    except OpenAIError as e:
        print(f"Error querying OpenAI API: {e}")
        return None


def login_to_system(username: str, password: str, login_url: str, answer: str) -> str | None:
    """Send login credentials and answer to the system."""
    payload = {
        "username": username,
        "password": password,
        "answer": answer
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        response = requests.post(login_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response
        else:
            print(f"Error: Received status code {response.status_code} from {login_url}")
    except requests.RequestException as e:
        print(f"Error: Failed to send login request due to {e}")
    return None


def fetch_secret_data(login_response: str) -> str | None:
    """
    Process the login_response from the secret page and extract relevant information.
    """
    # Save login_response content to a temporary HTML file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        f.write(login_response.text)
        temp_html_path = f.name
    
    # Open the temporary HTML file in the default web browser
    webbrowser.open(f'file://{temp_html_path}')

    # Extracts and prints the URL of the secret page
    print(f"Secret page URL: {login_response.url}")

    # Searches for and prints content enclosed in double curly braces (e.g., {{content}})
    matches = re.findall(r"{{(.*?)}}", login_response.text)
    for match in matches:
        print(f"Found content: {{ {match} }}")
    return True


def main() -> None:
    """Main function to orchestrate the login process."""
    username = os.getenv("USERNAME_s01e01")
    password = os.getenv("PASSWORD_s01e01")
    login_url = os.getenv("LOGIN_URL_s01e01")

    question = fetch_question(login_url)
    if not question:
        print("Failed to fetch the question.")
        return
    print(f"Question fetched: {question}")

    answer = get_answer_from_llm(question)
    if not answer:
        print("Failed to get an answer from the LLM.")
        return
    print(f"Answer: {answer}")

    login_response = login_to_system(username, password, login_url, answer)
    if not login_response:
        print("Login failed.")
        return
    print("Login successful.")

    secret_data = fetch_secret_data(login_response)
    if not secret_data:
        print("Failed to fetch secret data.")
        return
    print(f"Secret data fetched successfully.")


if __name__ == "__main__":
    """
    Entry point of the script. Executes the main function and handles any unhandled exceptions.
    """
    try:
        main()
    except Exception as err:
        print(f"[Error] {err}", file=sys.stderr)
        sys.exit(1)
