#!/bin/bash

for i in {2..4}
do
  # Create a new screen session named "run_session_i" and execute run.py with --arg set to the current value of i
  screen -dmS "rs_run_$i" bash -c "python runner.py --v1 $i --ver 'rs'"
done

# chmod +x run_script.sh
# run_script.sh