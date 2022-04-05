#!/bin/bash

DELAY=60m
DELAY_INT=$(( 60 * 60 ))

while true; do
    echo 'Running...'; 
    time make -B output/Dashboard.html PERSONAL=true FAST=true; 
    echo "Done at $(date --iso-8601=seconds)!"; 
    read -t $DELAY_INT -p "Waiting for $DELAY, or enter..."; 
done
