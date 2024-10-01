#!/bin/bash

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate tudat-bundle

export PYTHONPATH="/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python":$PYTHONPATH

# Default to 4 processes if not provided
NUM_PROCESSES=${1:-4}
shift  # Remove the first argument, which is the number of processes

# Run mpiexec with the specified or default number of processes
mpiexec -n $NUM_PROCESSES python "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/HyperVelocityImpacts/detumbling_HIV_MPI.py" 2

