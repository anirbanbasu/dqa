---
title: dqa
emoji: 🐳
colorFrom: purple
colorTo: blue
sdk: docker
suggested_hardware: cpu-basic
disable_embedding: true
license: mit
---

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

### Package manager and dependencies

- Install [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).
- Install project dependencies by running `uv sync --all-groups`.

### Observability

- Install a self-hosted, [with Docker](https://arize.com/docs/phoenix/self-hosting/deployment-options/docker), Arize Phoenix observability system by running `docker run -d --restart unless-stopped -p 6006:6006 -p 4317:4317 --name arizephoenix -i -t arizephoenix/phoenix:latest`. Alternatively, use [Arize Phoenix cloud](https://arize.com/docs/phoenix/phoenix-cloud).

### Durable agents (optional)
- Install a self-hosted, ([with Docker](https://docs.prefect.io/v3/how-to-guides/self-hosted/server-docker)), Prefect server by running `docker run -p 4200:4200 -d --restart unless-stopped --name prefect prefecthq/prefect:3-latest -- prefect server start --host 0.0.0.0`. _This is only necessary for [Durable Agents](https://ai.pydantic.dev/durable_execution/overview/) connecting to a self-hosted Prefect server_. Note that Durable Agents support is experimental and may be dropped in the future.

### Dapr
- Install and configure Dapr to run [with docker](https://docs.dapr.io/operations/hosting/self-hosted/self-hosted-with-docker/).
- Run `dapr init` to initialise `daprd` and the relevant containers.

_If deployment over Kubernetes is desired then check [these deployment instructions](https://docs.dapr.io/operations/hosting/kubernetes/kubernetes-deploy/)_.

### Dockerised, with `dapr --slim`

_Coming soon._

## Configuration and environment variables

There are two main configuration files.
 - LLM configuration at `conf/llm.json`.
 - MCP configuration at `conf/mcp.json`.

There is already a default configuration provided for either of these.

There are Dapr related configuration files too.
 - The main Dapr application configuration at `dapr.yaml`. Change the various hosts and ports in this configuration if you want to use ports that are not the default ones.
 - Dapr telemetry configuration at `.dapr/config.yaml`.
 - Dapr hot-swappable component configuration files at `.dapr/components/`.

The following environment variables are _mandatory_.
 - `DQA_SECRETS_REDIS_HOST`: Use this to specify your Redis host that Dapr will use for its _statestore_ and _pubsub_ components, e.g., `localhost:6379` or `host.internal.docker:6379`.
 - `DQA_SECRETS_REDIS_USERNAME`: Use this to specify your Redis username that Dapr will use to connect to the Redis host specified above, e.g., `default`.
 - `DQA_SECRETS_REDIS_PASSWORD`: Use this to specify your Redis password that Dapr will use to connect to the Redis host specified above. The password for local installations is typically blank.

The following API keys are optional but maybe provided for additional functionality. If not provided, the corresponding functionality will not be available.

 - `OLLAMA_API_KEY`: The Ollama API key is required for MCP-based web search, web fetch and cloud hosted models on Ollama. MCP features include 2 tools. Create an API key from [your Ollama account](https://ollama.com/settings/keys).
 - `ALPHAVANTAGE_API_KEY`: The API key to MCP-based Alpha Vantage finance related functions. MCP features include 118 tools. Obtain your [free API key from Alpha Vantage](https://www.alphavantage.co/support/#api-key). Note that basic finance tools are available through the [yfinance MCP](https://github.com/narumiruna/yfinance-mcp) even if Alpha Vantage MCP is not loaded.
 - `TAVILY_API_KEY`: The API key for Tavily search and research services. MCP features include 4 tools. Get your API key [from your Tavily account](https://app.tavily.com/home).

The following environment variables are all optional.
 - `APP_LOG_LEVEL`: The general log level of the DQA app. Defaults to `INFO`.
 - `DQA_MCP_SERVER_TRANSPORT`, `FASTMCP_HOST` and `FASTMCP_PORT`: These specify the transport type, the host and port for the built-in MCP server of DQA. The default values are `stdio`, `localhost` and `8000` respectively.
 - `HTTPX_TIMEOUT`: This specifies the timeout value in seconds a HTTP client will wait for the server to respond before raising an error. This applies, for example, to the communication timeout between the web frontend or CLI frontend to the A2A endpoint(s). Default value is 120 (seconds).
 - `LLM_CONFIG_FILE` and `MCP_CONFIG_FILE`: These specify where the LLM and MCP configurations These default to `conf/llm.json` and `conf/mcp.json` respectively.
 - [Gradio environment variables](https://www.gradio.app/guides/environment-variables) to configure the DQA web app. However, MCP server (not to be confused with DQA's built-in MCP server), server-side rendering (SSR) mode, API, Progressive Web App (PWA), analytics and public sharing will be disabled, irrespective of what is specified through the environment variables.
 - `PREFECT_API_URL`: This can be used to specify the Prefect Cloud API URL (in which case, you must set the `PREFECT_API_KEY`, see details) or the local self-hosted API URL at `http://localhost:4200/api`. The default value is None, which _will turn off Durable Agents_!
 - `PHOENIX_COLLECTOR_ENDPOINT`: This is used to specify the collector endpoint of an Arize Phoenix installation. The default value is `http://localhost:6006`. If you specify a remote endpoint that requires an API key, such as Phoenix cloud then you must also specify the environment variable `PHOENIX_API_KEY` and also specify `OTEL_EXPORTER_OTLP_HEADERS` with the value `'Authorization=Bearer <your-phoenix-api-key>'`.
 - `BROWSER_STATE_SECRET`: This is the secret used by Gradio to encrypt the browser state data. The default value is `a2a_dapr_bstate_secret`.
 - `BROWSER_STATE_CHAT_HISTORIES`: This is the key in browser state used by Gradio to store the chat histories (local values). The default value is `a2a_dapr_chat_histories`.
 - `APP_DAPR_SVC_HOST` and `APP_DAPR_SVC_PORT`: The host and port at which Dapr actor service will listen on. These default to `127.0.0.1` and `32768`. Should you change these, you must change the corresponding information in `dapr.yaml`.
 - `APP_DAPR_PUBSUB_STALE_MSG_SECS`: This specifies how old a message should be on the Dapr publish-subscribe topic queue before it will be considered too old, and dropped. The default value is 60 seconds.
 - `APP_DAPR_PUBSUB_MEMORY_STREAM_BUFFER_SIZE`: This specifies the internal memory object stream buffer size that is used to receive messages through the Dapr publish-subscribe subscription and pass it on to the A2A executor. Default value is 65536.
 - `APP_DAPR_ACTOR_RETRY_ATTEMPTS`: This specifies the number of times an agent executor will try to invoke a method on an actor, if it fails to succeed. The default value is 3.
 - `DAPR_PUBSUB_NAME`: The configured name of the publish-subscribe component at `.dapr/components/pubsub.yaml`. Change this environment variable only if you change the corresponding pub-sub component configuration.
 - `APP_A2A_SRV_HOST` and `APP_MHQA_A2A_SRV_PORT`: The host and port at which A2A endpoint will be available. These default to `127.0.0.1` and `32770`. Should you change these, you must change the corresponding information in `dapr.yaml`.
 - `APP_MHQA_A2A_REMOTE_URL`: This environment variable can be used to specify the full remote URL including the protocol, i.e., `http` or `https` where the MHQA A2A endpoint is available. This is useful in a scenario where the web app is deployed on a machine that is different from where the MHQA A2A endpoint and Dapr service are. Default value is `None`.

## Usage

- Start the Dapr actor service and the A2A endpoints by running `./start_dapr_multi.sh`. (This will send the Dapr sidecar processes in the background.)
- Invoke the A2A agent using JSON-RPC by calling `uv run dqa-cli --help` to learn about the various skills-based A2A endpoint invocations.
- Or, start the Gradio web app by running `uv run dqa-web-app` and then browse to http://localhost:7860.
- Once done, stop the dapr sidecars by running `./stop_dapr_multi.sh`.
- Alternatively, you could also run `./unified_start.sh` to run the Dapr sidecars as well as the web app, which will be available at http://localhost:7860. Press Ctrl+C to abort the server and the unified runner script will also shutdown the Dapr sidecars.
- Further to exposing the web app on localhost, you could also call `./run_ngrok.sh` (which requires you to have `ngrok` setup, [see instructions](https://ngrok.com/download/)) with an optional parameter `--domain your-ngrok-FQDN` to make your app available through `ngrok` publicly at https://your-ngrok-FQDN/.
  - Note that a feature of exposing the app over `ngrok` is that the configured traffic policy for `ngrok` will require any user accessing the app to authenticate themselves using an OAuth provider (GitHub). This is necessary to transparently create a user namespace such that two users concurrently each naming a chat ID as `test-chat` (for instance) will _not_ have a chat ID collision because each chat will be effectively handled by a different actor ID in the namespace based on the underlying OAuth provider supplied user ID. Thus, `user1__test-chat` is different from `user2__test-chat`.
  - Also note that the same OAuth-authenticated user using more than one separate browsers will _not_ have the same list of chat IDs visible on each browser. This is because the chat IDs list is local to each browser. However, the user will be able to load a specific chat ID on one browser in another browser by manually adding that chat ID. Since those chat IDs are in their OAuth-authenticated user namespaces, an authenticated user will only be able to add their own chat ID, not someone else's.

## Tests and coverage

Run `./run_tests.sh` to execute multiple tests and obtain coverage information. The script can accept additional arguments (e.g., `-k` to filter specific tests), which will be passed to `pytest`.
