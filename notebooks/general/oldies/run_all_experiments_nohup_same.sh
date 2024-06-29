#!/bin/bash

# Run all files in provided config_dir with the SAME script,
# found in script_path, so from the same model class. 

# Path to shell script that should be run in nohup
running_shell="run_all_experiments_in_dir.sh"

# Pass keyword arguments
# -s script_path, path to pyhton script
# -c config_dir, directory of config.yaml files to be run
while getopts "s:c:" flag
    do
             case "${flag}" in
                    s) script_path=${OPTARG};;
                    c) config_dir=${OPTARG};;
             esac
    done

[ -e output.txt ] && rm output.txt
nohup sh $running_shell $config_dir $script_path  > output.txt < /dev/null &