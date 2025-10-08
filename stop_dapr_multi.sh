#!/bin/bash
rm -fR .dapr/logs/
dapr stop --run-file dapr.yaml
