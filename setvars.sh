#!/bin/bash

# Activate the virtual environment
source /app/venv/bin/activate

# Source the setvars.sh script
source /opt/intel/setvars.sh

# Execute the original command
exec "$@"