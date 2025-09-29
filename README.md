[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#) [![pytest](https://github.com/anirbanbasu/py-a2a-dapr/actions/workflows/uv-pytest.yml/badge.svg)](https://github.com/anirbanbasu/py-a2a-dapr/actions/workflows/uv-pytest.yml)

# py-a2a-dapr

The Agent2Agent (A2A) protocol is [described](https://a2a-protocol.org/latest/) as:
> The Agent2Agent (A2A) Protocol is an open standard developed by Google and donated to the Linux Foundation designed to enable seamless communication and collaboration between AI agents.

Dapr actors are described in the [documentation](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-overview/) as:
> The [actor pattern](https://en.wikipedia.org/wiki/Actor_model) describes actors as the lowest-level “unit of computation”. [...] Dapr includes a runtime that specifically implements the [Virtual Actor pattern](https://www.microsoft.com/research/project/orleans-virtual-actors/).

**py-a2a-dapr** is a template repository for developing Dapr managed Agent2Agent (A2A) systems in Python.

## Overview

The components and their interactions in py-a2a-dapr is shown in the figure below. The template exposed in this project helps construct A2A endpoints that invoke Dapr actors. The endpoints and the Dapr actors, both run as Dapr applications with their respective Dapr sidecars. The JSON-RPC clients (e.g., a web or a CLI application, or just `curl`) interact with the A2A endpoints and are oblivous to the underlying actors.

![img-components-overview](https://raw.githubusercontent.com/anirbanbasu/py-a2a-dapr/master/docs/images/components-overview.svg)

## Installation

- Install [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/).
- Install project dependencies by running `uv sync --all-groups`.
- Configure Dapr to run [with docker](https://docs.dapr.io/operations/hosting/self-hosted/self-hosted-with-docker/).
- Run `dapr init` to initialise `daprd` and the relevant containers.

## Usage

- Start the Dapr actor service and the A2A endpoints by running `./start_dapr_multi.sh`. (This will send the dapr sidecar processes in the background.)
- Invoke the A2A agent using JSON-RPC by calling `uv run a2a-client --help` to learn about the various skills-based A2A endpoint invocations.
- Or, start the Gradio web app by running `uv run web-app` and then browse to http://localhost:7860.
- Once done, stop the dapr sidecars by running `./stop_dapr_multi.sh`.

## Tests and coverage

Run `./run_tests.sh` to execute multiple tests and obtain coverage information. The script can accept additional arguments (e.g., `-k` to filter specific tests), which will be passed to `pytest`.
