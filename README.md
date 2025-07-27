[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#)

# DQA: Difficult Questions Attempted

<p align="right">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

## Overview

The DQA aka _difficult questions attempted_ project utilises large language model (LLM) agent(s) to perform _multi-hop question answering_ (MHQA). This project has been inspired by the tutorial [^1] and the article [^2], both by [Dean Sacoransky](https://www.linkedin.com/in/dean-sacoransky-6a671119a/).

[^1]: Sacoransky, D., 2024. Build a RAG agent to answer complex questions. IBM Developer Tutorial. [URL](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/).
[^2]: Sacoransky, D., 2025. Reasoning & Recursive Retrieval With Deepseek-r1, Tavily, and LangGraph. Medium article. [URL](https://medium.com/@deansaco/reasoning-recursive-retrieval-with-deepseek-r1-tavily-and-langgraph-79b3336731e2).

## Project status

Following is a table of some updates regarding the project status. Note that these do not correspond to specific commits or milestones.

| Date     |  Status   |  Notes or observations   |
|----------|:-------------:|----------------------|
| July 27, 2025 |  active |  LlamaIndex Workflows replaced LangChain/LangGraph _again_. Using [Agent Communication Protocol](https://agentcommunicationprotocol.dev/) (ACP) to expose agentic functionalities. Gradio interface now uses ACP REST API to communicate with agents. |
| June 7, 2025 |  active |  Changed package and project manager to `uv` from `poetry`. Changed LLM orchestration to LangChain/LangGraph from DSPy. Added supporting MCP server. |
| February 15, 2025 |  active |  Custom adapter added for Deepseek models.  |
| January 26, 2025 |  active |  LlamaIndex Workflows replaced by DSPy.  |
| September 21, 2024 |  active |  Workflows made selectable.  |
| September 13, 2024 |  active |  Low parameter LLMs perform badly in unnecessary self-discovery, query refinements and ReAct tool selections.  |
| September 10, 2024 |  active |  Query decomposition may generate unnecessary sub-workflows.  |
| August 31, 2024 |  active |  Using built-in ReAct agent.  |
| August 29, 2024 |  active |  Project started.  |


## Installation

The directory where you clone this repository will be referred to as the _working directory_ or _WD_ hereinafter.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). To install the project with its essential dependencies in a virtual environment, run the following in the _WD_. To install all non-essential dependencies, add the `--all-groups` flag to the following command.

```bash
uv sync
```

In addition to project dependencies, see the installation instructions of [Ollama](https://ollama.com/download). You can also install it on a separate machine. Download a [tool calling model of Ollama](https://ollama.com/search?c=tools) that you want to use, e.g., `llama3.1` or `mistral-nemo`.

## Environment variables

Following is a list of environment variables that can be used to configure the DQA application. All environment variables should be supplied as quoted strings. They will be interpreted as the correct type as necessary.

For environment variables starting with `GRADIO_`, See [Gradio documentation for environment variables](https://www.gradio.app/guides/environment-variables).

When running the provided MCP server, the following environment variables can be specified, prefixed with `FASTMCP_SERVER_`: `HOST`, `PORT`, `DEBUG` and `LOG_LEVEL`. See [key configuration options](https://gofastmcp.com/servers/fastmcp#key-configuration-options) for FastMCP. Note that `on_duplicate_` prefixed options specified as environment variables _will be ignored_.

| Variable |  [Default value] and description   |
|--------------|----------------|
| `DQA_MCP_CLIENT_CONFIG` | [`config/mcp-client.json`] The path to the config file for providing a set of MCP servers. |
| `DQA_USE_MCP` | [`True`] If this is set to True, then make sure that the MCP client configuration points to available MCP server(s). You may want to run the provided MCP server. See below for instructions regarding that. |
| `DQA_LLM_CONFIG` | [`config/chat-ollama.json`] The path to the config file for the Ollama LLM provider. |
| `DQA_MCP_SERVER_TRANSPORT` | [streamable-http] The acceptable options are `stdio`, `sse` or `streamable-http`. |
| `DQA_LOGGING_LEVEL` | [INFO] The logging level of DQA and related servers. This option **may not be in use**. |
| `DQA_ACP_HOST` | [127.0.0.1] The host to which the Agent Communication Protocol (ACP) standalone Asynchronous Server Gateway Interface (ASGI) server should bind to. |
| `DQA_ACP_PORT` | [8192] The port on which the ACP AGSI server should listen on. |
| `DQA_ACP_CLIENT_ACCESS_URL` | [http://localhost:8192] The URL to which the ACP client should try to connect. Note that a HTTPS URL can be provided if the ASGI server is configured to serve over HTTPS. |
| `DQA_BROWSER_STATE_ENCRYPTION_KEY` | [your-secret-key-here] A secret key to encrypt information stored in browser storage. Use a secure key in production and public deployments. |

### Example LLM config

An example configuration, for the file pointed to by `DQA_LLM_CONFIG`, using locally available Ollama is shown below.

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "llama3.1:latest",
    "request_timeout": 180,
    "thinking": false
  }
}
```

### Example MCP config

An example configuration, for the file pointed to by `DQA_MCP_CLIENT_CONFIG`, using locally available Ollama is shown below.

```json
{
  "builtin": {
    "transport": "stdio",
    "command": "uv",
    "args": [
      "run",
      "dqa-mcp"
    ],
    "env": {
      "DQA_MCP_SERVER_TRANSPORT": "stdio"
    }
  },
  "frankfurter": {
    "transport": "stdio",
    "command": "uv",
    "args": [
      "run",
      "frankfurtermcp"
    ]
  },
  "ddg-search": {
    "transport": "stdio",
    "command": "uvx",
    "args": [
      "duckduckgo-mcp-server"
    ]
  }
}
```

## Usage

Create a `.env` file in the _WD_, to set the environment variables as above, if you want to use anything other than the default settings. There is a template `.env.template` file to get you started.

### Optional DQA MCP server

Run the following in the _WD_ to start the MCP server. This is **not needed** to run the DQA application because the underlying agent workflow will automatically start the MCP server in `stdio` mode.

```bash
uv run dqa-mcp
```

### DQA app as an ASGI server

Run the following in the _WD_ to start the DQA app as a standalone ASGI server.

```bash
uv run dqa-acp
```
If using the default values, the server will be available at [http://localhost:8192](http://localhost:8192) with API documentation at [http://localhost:8192/docs](http://localhost:8192/docs).

### DQA webapp

Run the following in the _WD_ to start the web server.

```bash
uv run dqa-webapp
```
The web UI will be available at [http://localhost:7860](http://localhost:7860). To exit the server, use the Ctrl+C key combination.

### DQA ACP command-line interface (CLI) client

Run the following in the _WD_ to start access the **experimental** command-line interface. In addition to the aforementioned environment variables, you can specify `DQA_SESSION_ID` to make the CLI client to attach to an existing ACP session.

```bash
uv run dqa-acp-client
```

## Contributing

Install developer tools by running `uv sync --all-groups`. Then, install the `pre-commit` hooks by running the following.

```bash
pre-commit install
```
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/).
