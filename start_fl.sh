#!/bin/bash

nvflare simulator \
  -w $PWD/workspace \
  -c liver,spleen,pancreas,kidney \
  -gpu 0,1,2,4 \
  -n 4 \
  -t 1 \
  apps
