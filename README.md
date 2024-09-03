[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#)
[![Python linter and format checker with ruff](https://github.com/anirbanbasu/dqa/actions/workflows/python-linter-format-checker.yml/badge.svg)](https://github.com/anirbanbasu/dqa/actions/workflows/python-linter-format-checker.yml)

# DQA: Difficult Questions Attempted

<p align="right">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

The DQA aka _difficult questions attempted_ project aims to make large language models attempt difficult questions through an agent-based architecture. The project utilises retrieval augmented generation (RAG) where relevant. The project is inspired by a tutorial [^1] from [Dean Sacoransky](https://www.linkedin.com/in/dean-sacoransky-6a671119a/).

[^1]: Sacoransky, D., 2024. Build a RAG agent to answer complex questions. IBM Developer Tutorial. [URL](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/).

## Project status

 - August 31, 2024: Actively maintained experimental prototype.

## Installation

Install and activate a Python virtual environment in the directory where you have cloned this repository. Let us refer to this directory as the _working directory_ or _WD_ (interchangeably) hereonafter. You could do that using [pyenv](https://github.com/pyenv/pyenv), for example. Make sure you use Python 3.12.0 or later. Inside the activated virtual environment, run the following.

```bash
python -m pip install -U pip
python -m pip install -U -r requirements.txt
```

In addition to Python dependencies, see the installation instructions of [Ollama](https://ollama.com/download) and that of [Qdrant](https://qdrant.tech/documentation/guides/installation/). You can install either of these on separate machines. Download the [tool calling model of Ollama](https://ollama.com/search?c=tools) that you want to use, e.g., `llama3.1` or `mistral-nemo`.

## Usage (local)

Make a copy of the file `.env.docker` in the _working directory_ as a `.env` file.

```bash
cp .env.docker .env
```

Change all occurrences of `host.docker.internal` to `localhost` or some other host or IP assuming that you have both Ollama and Qdrant available at ports 11434 and 6333, respectively, on your preferred host. Set the Ollama model to the tool calling model that you have downloaded on your Ollama installation. Set the value of the `LLM_PROVIDER` to the provider that you want to use. Supported names are `Anthropic`, `Cohere`, `Groq`, `Ollama` and `Open AI`.

You can use the environment variable `SUPPORTED_LLM_PROVIDERS` to further restrict the supported LLM providers to a subset of the aforementioned, such as, by setting the value to `Groq:Ollama` to allow only Groq and Ollama for some deployment of this app. Note that the only separating character between LLM provider names is a `:`. If you add a provider that is not in the aforementioned set, the app will throw an error and refuse to start.

Add the API keys for [Anthropic](https://console.anthropic.com/), [Cohere](https://dashboard.cohere.com/welcome/login), [Groq](https://console.groq.com/keys) or [Open AI](https://platform.openai.com/docs/overview) if you want to use any of these. In addition, add [an API key of Tavily](https://app.tavily.com/sign-in). Qdrant API key is not necessary if you are not using [Qdrant cloud](https://qdrant.tech/documentation/qdrant-cloud-api/).

With all these setup done, run the following to start the web server. The web server will serve a web user interface as well as a REST API. It is not configured to use HTTPS.

```bash
python src/webapp.py
```

If you want to see the experimental console log on the web UI, run the following command instead. Do remember to delete `/tmp/dqa.log` when you no longer need it.

```bash
python src/webapp.py 2>&1 | tee /tmp/dqa.log
```

The web UI will be available at [http://localhost:7860](http://localhost:7860).

## Usage (Docker)

In the `.env.docker`, both Ollama and Qdrant are expected to be available at ports 11434 and 6333, respectively, on your Docker host, i.e., `host.docker.internal`. Set them to some other host(s), if that is where your Ollama and Qdrant servers are available. Set the Ollama model to the tool calling model that you have downloaded on your Ollama installation.

Set the value of the `LLM_PROVIDER` to the provider that you want to use and add the API keys for Anthropic, Cohere, Groq and Open AI LLM providers as well as that of Tavily and optionally Qdrant as metioned above in the **Usage (local)** section.

With all these setup done, and assuming that you have Docker installed, you can build an image of the DQA app, create a container and start it as follows.

```bash
docker build -f local.dockerfile -t dqa .
docker create -p 7860:7860/tcp --name dqa-container dqa
docker container start dqa-container
```

You can replace the second line above to the following, in order to use a `.env` file on your Docker host that resides at the absolute path `PATH_TO_YOUR_.env_FILE`.

```bash
docker create -v /PATH_TO_YOUR_.env_FILE:/home/app_user/app/.env -p 7860:7860/tcp --name dqa-container dqa
```

The web server will serve a web user interface as well as a REST API at [http://localhost:7860](http://localhost:7860). It is not configured to use HTTPS.

## Contributing

Install [`pre-commit`](https://pre-commit.com/) for Git and [`ruff`](https://docs.astral.sh/ruff/installation/). Then enable `pre-commit` by running the following in the _WD_.

```bash
pre-commit install
```
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
