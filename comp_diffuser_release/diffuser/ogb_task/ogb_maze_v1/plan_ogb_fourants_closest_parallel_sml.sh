#!/bin/bash

# Thin wrapper: closest-ant parallel mode is USE_PARALLEL=1 on the unified script.
exec bash diffuser/ogb_task/ogb_maze_v1/plan_ogb_fourants_sml.sh "$1" "$2" "$3" 1 "${4:-1}"
