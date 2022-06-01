#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUT_SUFFIX="u2g.out"
RUN_SCRIPT="run_base.sh"

for ((j=5;j<=90;j+=5)); do
    export LMBD=$j
    export NAME=bash
    export OUTNAME=${OUT_SUFFIX}
    sbatch --job-name=${NAME} \
    --output=${OUTNAME} \
    --exclude=gonzo \
    ${RUN_SCRIPT}
done



            
