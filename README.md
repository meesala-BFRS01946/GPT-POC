# FAQ Answering API

This is a Flask-based API that acts as an FAQ answerer for an e-commerce company. It provides answers to user questions based on a pre-trained model and a labeled dataset.

## Requirements

- Python 3.x
- Flask
- Redis
- spaCy
- FuzzyWuzzy
- OpenSearch
- dotenv

## Installation

1. Clone the repository:

2. Create a virtual environment:

3. Install the dependencies:

4. Set the environment variables:
- Rename the `.env.example` file to `.env`.
- Add your OpenAI API key to the `OPENAI_API_KEY` variable in the `.env` file.

5. Prepare the dataset:
- Place your labeled dataset file (e.g., `Questions.csv`) in the project directory.

6. Start the API server:

## API Usage

Send a POST request to the root endpoint (`/`) with a JSON payload containing the user question:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"How can I track my order?"}' http://localhost:5000/
{
  "response": "<answer_html>",
  "intent": "openai_api"
}
