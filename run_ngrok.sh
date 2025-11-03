#!/bin/bash
# Add a domain such as --domain your-custom-domain.ngrok-free.app
# Add --traffic-policy-file ngrok_traffic_policy.yml if your ngrok plan supports it
ngrok http 7860 --host-header='localhost:7860' --name 'dqa-ngrok' --log-format 'json' --traffic-policy-file ngrok_traffic_policy.yml $@
