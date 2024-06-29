#!/bin/bash

# Run all files in provided config_dir with the SAME script 
# found in script_path

# Path to shell script that should be run in nohup
running_shell="./scripts/general/run_all_experiments_in_dir.sh"

# Pass keyword arguments
# -s script_path, path to pyhton script
# -c config_dir, directory of config.yaml files to be run
while getopts "s:c:" flag
    do
             case "${flag}" in
                    c) config_dir=${OPTARG};;
             esac
    done

[ -e output.txt ] && rm output.txt
nohup sh $running_shell $config_dir > output.txt < /dev/null &