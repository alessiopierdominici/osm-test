### Training remoto, usare questo comando per fare il training remoto
```
nohup python3 finetuning-4-datasets.py > my_output_finetuning-4-datasets_cuda.log 2>&1 &

nohup python3 eval.py > my_output.log 2>&1 &

nohup python3 train-OSM.py > my_output_train-OSM_cuda0.log 2>&1 &

nohup python3 train-OSM-gradient-accumulation.py > my_output_train-OSM-gradient-accumulation_cuda1_bs256.log 2>&1 &

nohup python3 combined-ft-osm.py > my_output_combined-ft-osm_cuda0.log 2>&1 &
```

To stop a background process started with nohup, you can use the kill command with the PID of the process, as I mentioned earlier. Here's how to do it:

First, you need to find the PID of the Python process that you want to stop. You can do this using the ps command:
```
ps aux | grep my_script.py

ps aux | grep finetuning-4-datasets.py

ps aux | grep train-OSM.py

ps aux | grep combined-ft-osm.py
```
This will display a list of processes matching the name my_script.py, along with their PIDs.

Identify the PID of the Python process you want to stop from the output of the ps command.
Once you have the PID, you can use the kill command to terminate the process:
```
kill PID
```
Replace PID with the actual PID of your Python process.

This will send a termination signal to the process, causing it to stop running. If the process is not terminated immediately, you can use the -9 option with kill to forcefully terminate it:
```
kill -9 PID
```
This should stop the process immediately, but it's generally recommended to try terminating the process gracefully first with just kill PID.