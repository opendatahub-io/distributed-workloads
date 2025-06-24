#!/bin/bash

set -o allexport
source .env-odh
set +o allexport

gotestsum "$@"