#!/bin/bash
# This script gets the file paths associated with a samdef, and hadds them into a single file.

SAMDEF=$1

# Clear the output file if it exists and rewrite the file names to it
> temp_filelist.list
samweb list-files "defname:$SAMDEF" > temp_filelist.list

# # Check if the sample is prestaged
# samweb prestage-dataset --defname=$SAMDEF

# Go through temp_filelist and get the path of each file
> filelist.list
while read -r line; do
    # Check if the line is not empty
    FILEPATH=$(samweb locate-file $line)
    # Remove leading enstore: from the file path
    FILEPATH=$(echo "$FILEPATH" | sed 's/^enstore://')
    FILEPATH=$(echo "$FILEPATH" | sed 's/([^)]*)//g')
    if [[ -n "$line" ]]; then
        echo "$FILEPATH/$line" >> filelist.list
    fi
done < temp_filelist.list

hadd /exp/uboone/data/users/jbateman/workdir/HPS_uboone_analysis/Run_245_ana/"$SAMDEF.root" @filelist.list
