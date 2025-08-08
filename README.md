# ai3r2

This repository contains a collection of projects completed as part of the [AI_devs3 Reloaded](https://www.aidevs.pl/) course, focused on the practical development of autonomous AI agents and Generative AI tools. The course covers working with large language models (LLMs), API integration, RAG techniques, multimodality (text, image, voice), and building custom AI systems to solve real-world problems.

## Project Structure

The repository is divided into thematic projects, following the training schedule (five weeks (season), where each week contains five days (episode)):

### Season 1
- `E01` This script automates the login process to a system that requires answering a dynamic question.
It uses the OpenAI API to generate an answer based on the question fetched from the login page. 
No additional web search is required as the LLM can answer the question directly, as the questions 
are based on historical events.

- `E02` This script automates a conversation with an external API and OpenAI's GPT model. It sends an initial POST request to start the conversation, queries OpenAI's GPT model for an answer to the received question, and later sends the answer back to the API to retrieve a required phrase. All to imitate a robot-robot interaction.

## Requirements

The projects use various Python libraries listed in the `requirements.txt` file. To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Authors

KCzenczek and the tiny but mighty friend <img src="other_files/github-copilot-icon.svg" alt="Copilot Icon" style="width:50px; height:50px;">