#! /bin/bash

sbatch --requeue exp-MH.sub
sbatch --requeue exp-AM.sub
sbatch --requeue exp-cKAM.sub
sbatch --requeue exp-GAM.sub
sbatch --requeue exp-KAM.sub
sbatch --requeue exp-RBAM.sub

