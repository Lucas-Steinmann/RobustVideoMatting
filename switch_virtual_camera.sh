#!/usr/bin/env bash
# Script to start and stop the webcam loop.
# Assumes the background image at the following relative path...
background="background.png"
# ... as well as a virtual env at the following relative path
venv=".venv/bin/activate"

# Source: https://stackoverflow.com/a/246128/13182493
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if pgrep -f "python3 ./webcam";
then
	pkill -f "python3 ./webcam"
else
	cd $SCRIPT_DIR
	source $venv
	./webcam.py -b $background >> webcam.log 2>&1 & disown
fi
