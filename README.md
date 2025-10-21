# Foodi_AIAgent

Multigoo is a modular, multi-agent AI system built on LangGraph and LangChain.
It plans, searches, and reasons across tools to answer complex food-related questions.

This project implemented using an LLMs Compiler architecture that automatically generates and executes tool plans in parallel.
This ensures that web search, YouTube retrieval, and nutrition estimation run concurrently for maximum efficiency and lower LLM cost.

## Requirements

- Python 3.12 or later

### Install Python using MiniConda

1. Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2. Create a new environment:
   ```bash
   conda create -n mini-rag python=3.8
3) Activate the environment:
    ```bash
    $ conda activate mini-rag
   
## Installation

### Install the required dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environments.

To install all required packages, run:

```bash
# Install dependencies from pyproject.toml
poetry install
```

If this is your first time using Poetry, you can install it with:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```


## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials



```bash
$ cd docker
$ sudo docker compose up -d
```

## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## POSTMAN Collection

Download the POSTMAN collection from [/assets/mini-rag-app.postman_collection.json](/assets/mini-rag-app.postman_collection.json)