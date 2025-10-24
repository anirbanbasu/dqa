#!/bin/bash
./start_dapr_multi.sh
uv run dqa-web-app
./stop_dapr_multi.sh
