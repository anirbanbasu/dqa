#!/bin/bash
# Ensure any existing Dapr instances are stopped
dapr stop --run-file dapr.yaml
# Start Dapr with the specified configuration file
dapr run --run-file dapr.yaml &
