[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#) [![pytest](https://github.com/anirbanbasu/dqa/actions/workflows/uv-pytest.yml/badge.svg)](https://github.com/anirbanbasu/dqa/actions/workflows/uv-pytest.yml)

# DQA: Difficult Questions Attempted

<p align="center">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/docs/images/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

The DQA aka _difficult questions attempted_ project utilises large language model (LLM) agent(s) to perform _multi-hop question answering_ (MHQA).

_Note that this repository is undergoing a complete overhaul of the [older, now obsolete, version of DQA](https://github.com/anirbanbasu/dqa-obsolete). The purpose of this overhaul is to standardise agent communication using the A2A protocol and to use the Dapr virtual actors to manage the backend logic._

## Overview
DQA - Difficult Questions Attempted - is an agentic chatbot that attempts to answer non-trivial multi-hop questions using (large) language models (LLM) and tools available over the Model Context Protocol (MCP). The functionality of DQA is basic. As of late 2025, the functionality of DQA is available in most commercial chatbots.

However, DQA is experimental with the emphasis on standardising agentic communication and managing backend functionality using Dapr-managed virtual actors. Although the instructions below explain the deployment of DQA on a single machine, it can be deployed and run on a Kubernetes cluster, with minimal modifications to the configuration.

## Installation

- Install [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).
- Install project dependencies by running `uv sync --all-groups`.
- Configure Dapr to run [with docker](https://docs.dapr.io/operations/hosting/self-hosted/self-hosted-with-docker/).
- Run `dapr init` to initialise `daprd` and the relevant containers.

_If deployment over Kubernetes is desired then check [these deployment instructions](https://docs.dapr.io/operations/hosting/kubernetes/kubernetes-deploy/)_.

## Configuration and environment variables

There are two main configuration files.
 - LLM configuration at `conf/llm.json`.
 - MCP configuration at `conf/mcp.json`.

There is already a default configuration provided for either of these.

There are Dapr related configuration files too.
 - The main Dapr application configuration at `dapr.yaml`. Change the various hosts and ports in this configuration if you want to use ports that are not the default ones.
 - Dapr telemetry configuration at `.dapr/config.yaml`.
 - Dapr hot-swappable component configuration files at `.dapr/components/`.



The following API keys are optional but maybe provided for additional functionality. If not provided, the corresponding functionality will not be available.

 - `OLLAMA_API_KEY`: The Ollama API key is required for MCP-based web search, web fetch and cloud hosted models on Ollama. MCP features include 2 tools. Create an API key from [your Ollama account](https://ollama.com/settings/keys).
 - `ALPHAVANTAGE_API_KEY`: The API key to MCP-based Alpha Vantage finance related functions. MCP features include 118 tools. Obtain your [free API key from Alpha Vantage](https://www.alphavantage.co/support/#api-key). Note that basic finance tools are available through the [yfinance MCP](https://github.com/narumiruna/yfinance-mcp) even if Alpha Vantage MCP is not loaded.
 - `TAVILY_API_KEY`: The API key for Tavily search and research services. MCP features include 4 tools. Get your API key [from your Tavily account](https://app.tavily.com/home).

The following environment variables are all optional.
 - `APP_LOG_LEVEL`: The general log level of the DQA app. Defaults to `INFO`.
 - `DQA_MCP_SERVER_TRANSPORT`, `FASTMCP_HOST` and `FASTMCP_PORT`: These specify the transport type, the host and port for the built-in MCP server of DQA. The default values are `stdio`, `localhost` and `8000` respectively.
 - `LLM_CONFIG_FILE` and `MCP_CONFIG_FILE`: These specify where the LLM and MCP configurations These default to `conf/llm.json` and `conf/mcp.json` respectively.
 - [Gradio environment variables](https://www.gradio.app/guides/environment-variables) to configure the DQA web app. However, MCP server (not to be confused with DQA's built-in MCP server), SSR mode, API and public sharing will be disabled, irrespective of what is specified through the environment variables.
 - `BROWSER_STATE_SECRET`: This is the secret used by Gradio to encrypt the browser state data. The default value is `a2a_dapr_bstate_secret`.
 - `BROWSER_STATE_CHAT_HISTORIES`: This is the key in browser state used by Gradio to store the chat histories (local values). The default value is `a2a_dapr_chat_histories`.
 - `APP_DAPR_SVC_HOST` and `APP_DAPR_SVC_PORT`: The host and port at which Dapr actor service will listen on. These default to `127.0.0.1` and `32768`. Should you change these, you must change the corresponding information in `dapr.yaml`.
 - `APP_DAPR_PUBSUB_STALE_MSG_SECS`: This specifies how old a message should be on the Dapr publish-subscribe topic queue before it will be considered too old, and dropped. The default value is 60 seconds.
 - `APP_DAPR_ACTOR_RETRY_ATTEMPTS`: This specifies the number of times an agent executor will try to invoke a method on an actor, if it fails to succeed. The default value is 3.
 - `DAPR_PUBSUB_NAME`: The configured name of the publish-subscribe component at `.dapr/components/pubsub.yaml`. Change this environment variable only if you change the corresponding pub-sub component configuration.
 - `APP_A2A_SRV_HOST` and `APP_MHQA_A2A_SRV_PORT`: The host and port at which A2A endpoint will be available. These default to `127.0.0.1` and `32770`. Should you change these, you must change the corresponding information in `dapr.yaml`.
 - `APP_MHQA_A2A_REMOTE_URL`: This environment variable can be used to specify the full remote URL including the protocol, i.e., `http` or `https` where the MHQA A2A endpoint is available. This is useful in a scenario where the web app is deployed on a machine that is different from where the MHQA A2A endpoint and Dapr service are. Default value is `None`.

## Usage

- Start the Dapr actor service and the A2A endpoints by running `./start_dapr_multi.sh`. (This will send the Dapr sidecar processes in the background.)
- Invoke the A2A agent using JSON-RPC by calling `uv run dqa-cli --help` to learn about the various skills-based A2A endpoint invocations.
- Or, start the Gradio web app by running `uv run dqa-web-app` and then browse to http://localhost:7860.
- Once done, stop the dapr sidecars by running `./stop_dapr_multi.sh`.

## Tests and coverage

Run `./run_tests.sh` to execute multiple tests and obtain coverage information. The script can accept additional arguments (e.g., `-k` to filter specific tests), which will be passed to `pytest`.
