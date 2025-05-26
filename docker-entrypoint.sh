#!/bin/bash

    # Source the setvars.sh script
    source /opt/intel/setvars.sh

    # Execute the original command
    exec "$@"
