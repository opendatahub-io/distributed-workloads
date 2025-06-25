#!/bin/bash

set -o allexport
source .env-rhoai
set +o allexport

gotestsum "$@"