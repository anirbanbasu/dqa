#!/bin/bash
trap './stop_dapr_multi.sh' EXIT INT TERM
./start_dapr_multi.sh
uv run dqa-web-app
# ./stop_dapr_multi.sh
