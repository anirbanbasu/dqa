#!/bin/bash
# Clean up old logs
rm -fR .dapr/logs/
# Ensure any existing Dapr instances are stopped
dapr stop --run-file dapr.yaml
# Start Dapr with the specified configuration file
dapr run --run-file dapr.yaml &
