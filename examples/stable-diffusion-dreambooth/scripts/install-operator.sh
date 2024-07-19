#!/bin/sh

NAMESPACE=$1
DISPLAY_NAME=$2

get_csv_name() {
  oc get csv -n "$NAMESPACE" -o jsonpath="{.items[?(@.spec.displayName==\"$DISPLAY_NAME\")].metadata.name}"
}

# Initial CSV name check
csv_name=$(get_csv_name)

# Loop to wait until the CSV is created
while [ -z "$csv_name" ]; do
  echo "Waiting for $DISPLAY_NAME CSV to be created in namespace $NAMESPACE..."
  sleep 10
  csv_name=$(get_csv_name)
done

# Loop to wait until the CSV phase is "Succeeded"
until oc get csv "$csv_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' | grep -xq "Succeeded"; do
  current_phase=$(oc get csv "$csv_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
  echo "Current Phase: $current_phase"
  echo "Waiting for $DISPLAY_NAME to be ready in namespace $NAMESPACE..."
  sleep 10
done

echo "$DISPLAY_NAME is installed"
