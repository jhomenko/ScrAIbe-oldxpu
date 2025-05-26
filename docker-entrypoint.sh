#!/bin/bash

    # Source the setvars.sh script
    source /opt/intel/oneapi/setvars.sh

    # Execute the original command
    exec "$@"