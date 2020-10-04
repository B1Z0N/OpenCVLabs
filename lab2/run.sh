#!/bin/sh

OWNERS="anton nikolay"
OUTPUT_DIR="output"

mkdir -p $OUTPUT_DIR

for code_owner in $OWNERS
do
    for media_owner in $OWNERS 
    do 
	    printf "\n\nCode is from %s, media is from %s\n\n" $code_owner $media_owner
	    CODE_OWNER=$code_owner MEDIA_OWNER=$media_owner python3 solution.py > "${OUTPUT_DIR}/${code_owner}-code_${media_owner}-media_.json"
    done 
done
