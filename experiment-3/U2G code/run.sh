#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUT_SUFFIX="u2g.out"
RUN_SCRIPT="run_base.sh"

for i in $(seq -4.605 0.2 -5); do
    export LMBD=$i
    export NAME=bash
    export OUTNAME=${OUT_SUFFIX}
    sbatch --job-name=${NAME} \
    --output=${OUTNAME} \
    --exclude=gonzo \
    ${RUN_SCRIPT}
done



            
