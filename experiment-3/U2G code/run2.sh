#!/bin/bash
#

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUT_SUFFIX="u2g.out"
RUN_SCRIPT="run_base2.sh"

for ((j=1;j<=30;j+=1)); do
    export LMBD=$j
    export NAME=bash
    export OUTNAME=${OUT_SUFFIX}
    sbatch --job-name=${NAME} \
    --output=${OUTNAME} \
    --exclude=gonzo \
    ${RUN_SCRIPT}
done



            
