#!/bin/bash

for i in {0..4}
do
  # Create a new screen session named "run_session_i" and execute run.py with --arg set to the current value of i
  screen -dmS "run_$i" bash -c "python runner.py --v1 $i"
done

# chmod +x run_script.sh
# run_script.sh