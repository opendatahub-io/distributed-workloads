#!/bin/bash
OPENSHIFT_GOIMPORTS=${1}

# ${OPENSHIFT_GOIMPORTS} -l
bad_files=$(${OPENSHIFT_GOIMPORTS} -l)

echo $bad_files
if [[ -n ${bad_files} ]]; then
        echo "!!! openshift-goimports needs to be run on the following files:"
        echo "${bad_files}"
        echo "Try running 'make imports'"
        exit 1
fi