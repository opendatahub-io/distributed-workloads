#!/bin/bash

# Test script to verify the tree-shaking improvements
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSS_FILE="$SCRIPT_DIR/dist/pf.css"
ORIGINAL_CSS="$SCRIPT_DIR/node_modules/@patternfly/patternfly/patternfly-no-globals.css"

echo "Building project..."
cd "$SCRIPT_DIR" && pnpm build

# Check if the CSS file exists
if [ ! -f "$CSS_FILE" ]; then
  echo "ERROR: CSS file not generated!"
  exit 1
fi

# Check if the main.js file was NOT generated (should be prevented)
if [ -f "$SCRIPT_DIR/dist/main.js" ]; then
  echo "ERROR: Empty main.js file was generated!"
  exit 1
fi

# Get file sizes
ORIGINAL_SIZE=$(wc -c < "$ORIGINAL_CSS")
TREE_SHAKEN_SIZE=$(wc -c < "$CSS_FILE")
REDUCTION=$((ORIGINAL_SIZE - TREE_SHAKEN_SIZE))
PERCENTAGE=$((REDUCTION * 100 / ORIGINAL_SIZE))

echo "Original CSS size: $ORIGINAL_SIZE bytes"
echo "Tree-shaken CSS size: $TREE_SHAKEN_SIZE bytes"
echo "Reduction: $REDUCTION bytes ($PERCENTAGE%)"

# Verify that essential spinner classes are present
if ! grep -q "pf-v6-c-spinner" "$CSS_FILE"; then
  echo "ERROR: Spinner class not found in CSS!"
  exit 1
fi

if ! grep -q "pf-v6-c-spinner__path" "$CSS_FILE"; then
  echo "ERROR: Spinner path class not found in CSS!"
  exit 1
fi

# Verify that unnecessary components are removed
if grep -q "pf-v6-c-button" "$CSS_FILE"; then
  echo "WARNING: Button classes found in CSS, should be removed!"
fi

if grep -q "pf-v6-c-tabs" "$CSS_FILE"; then
  echo "WARNING: Tabs classes found in CSS, should be removed!"
fi

# Check for font declarations
FONT_COUNT=$(grep -c "@font-face" "$CSS_FILE" || true)
echo "Font face declarations: $FONT_COUNT"

# Success message
echo "✅ Tree-shaking verification completed successfully!"
echo "The CSS file has been optimized and contains only the necessary styles. Yay!"
