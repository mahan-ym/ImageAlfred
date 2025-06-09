#!/bin/bash

# Set variables
REPO_URL="https://github.com/mahan-ym/ImageAlfred"
REPO_DIR="ImageAlfred"
TEMP_DIR="./tmp"
SRC_DIR="src"
REQUIREMENTS_FILE="requirements.txt"

echo "🚀 Starting Huggingface Space update script..."

# Clone or update the repository
ORIGINAL_DIR=$(pwd)
if [ -d "$TEMP_DIR" ]; then
    echo "📥 Updating repository..."
    cd "$TEMP_DIR" && git pull
else
    echo "📥 Cloning repository..."
    mkdir -p "$TEMP_DIR" && cd "$TEMP_DIR" && git clone "$REPO_URL"
fi
cd "$ORIGINAL_DIR"  # Return to original directory

# Copy src directory to current directory
echo "📁 Updating source code..."
if [ ! -d "$TEMP_DIR/$REPO_DIR/$SRC_DIR" ]; then
    echo "❌ Source directory not found in the repository!"
    exit 1
fi

if [ -d "$SRC_DIR" ]; then
    rm -rf "$SRC_DIR"
fi
cp -r "$TEMP_DIR/$REPO_DIR/$SRC_DIR" .
mv "$TEMP_DIR/$REPO_DIR/Makefile" .

# Check if copy was successful
if [ $? -eq 0 ]; then
    rm -rf "$TEMP_DIR"
    echo "✅ Source code updated successfully!"
else
    echo "❌ Failed to copy source code!"
    exit 1
fi

echo "🎉 Update completed! Source code and requirements are now up to date."