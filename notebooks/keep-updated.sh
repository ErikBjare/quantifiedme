#!/bin/bash

DELAY=15m
DELAY_INT=$(( $(echo $DELAY | grep -oP '[0-9]+') * 60 ))

while true; do
    echo 'Running...'; 
    time make -B output/Dashboard.html PERSONAL=true FAST=true; 
    notify-send -a 'Job' 'Build successful'
    echo "Done at $(date --iso-8601=seconds)!"; 
    read -t $DELAY_INT -p "Waiting for $DELAY, or enter..."; 
done
