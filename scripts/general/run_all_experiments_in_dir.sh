#!/bin/bash

# Note: Experiment folder will be created in 
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}


# Save current wd 
cwd=$(pwd)

# Save config_dir and script_path
config_dir=$1

# Go to directory of configs 
cd $config_dir
# Loop over all files with extension .yaml in specified dir
for i in *.yaml; do
   # Read config
   eval $(parse_yaml $i)

   # Make new dir for experiment_class/exp_name
   # Move and rename config file there
   cd $cwd
    
   folder="$exp_class"/"$exp_name"
   mkdir -p experiments/"$folder"
   mv --force "$config_dir"/"$i" experiments/"$folder"/"$exp_name".yaml

   echo -e "\nExperiment directory for $exp_name created.\n" 
   
   echo -e "Starting experiment: $exp_name\n"

   # Run python script and pass config
   python $script_path -f experiments/"$folder"/"$exp_name".yaml

   cd $config_dir 

   echo -e "######################################################\n\n\n"

done



