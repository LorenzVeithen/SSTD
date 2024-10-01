#!/bin/bash

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate tudat-bundle

export PYTHONPATH="/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python":$PYTHONPATH

NUM_PROCESSES=${1:-1}
shift  # Remove the first argument, which is the number of processes

# Run mpiexec with the specified or default number of processes
mpiexec -n $NUM_PROCESSES python "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/LongTermTumblingAnalysis/LTT_sun_pointing_MPI.py" 2 0

