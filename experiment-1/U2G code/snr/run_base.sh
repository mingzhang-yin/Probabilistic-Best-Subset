#!/bin/sh
#
#SBATCH -A sml
#SBATCH --cpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH --nodelist=statler,waldorf
##SBATCH --mail-user=mzyin@utexas.edu
##SBATCH --mail-type=END

echo "python u2g.py  --lmbd ${LMBD}"
 
python u2g.py --lmbd ${LMBD}


