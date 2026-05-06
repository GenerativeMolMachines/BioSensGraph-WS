#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No input file specified."
    exit 1
fi

if [ ! -f $1 ]; then
    echo "File $1 does not exist."
    exit 1
fi

awk -F 'Stats:' '{
            n = split($2, a, ",")
                printf "{"
                for (i=1; i<=n; i++) {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", a[i])
                    split(a[i], kv, ":")
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", kv[1])
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", kv[2])
                    printf "\"%s\": %s", kv[1], kv[2]
                    if (i<n) printf ", "
                }
                print "}"
            }' $1 | grep loss
