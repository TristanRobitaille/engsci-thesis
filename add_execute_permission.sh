#!/bin/bash
# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY=$1

# Find and add execute permission to all .sh files
find "$DIRECTORY" -type f -name "*.sh" -exec chmod +x {} \;

echo "Execute permissions added to all .sh files in $DIRECTORY and its subdirectories."