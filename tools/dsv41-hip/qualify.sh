#!/bin/bash
# qualify.sh: the port's ordered mandatory gates and one verdict (handoff section 10); see qualify.py for the steps.
#   tools/dsv41-hip/qualify.sh [--conf CONF] [--out DIR] [--steps LIST] [--long 0|1] [--plan]
exec python3 "$(dirname "$0")/qualify.py" "$@"
