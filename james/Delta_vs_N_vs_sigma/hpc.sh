#!/bin/bash

# --- SLURM Settings ---
#SBATCH --job-name=JuliaDMRG
#SBATCH --time=13-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G          
#SBATCH --partition=compute
#SBATCH --output=output.log
#SBATCH --account=phys025062

# --- Notifications ---
#SBATCH --mail-user=xu22252@bristol.ac.uk
#SBATCH --mail-type=END,FAIL

# --- Your Commands ---
echo "Job started at $(date)"
echo "Using $SLURM_CPUS_PER_TASK CPU cores on node $SLURM_NODELIST"
echo "Requested $SLURM_MEM_PER_NODE MB total memory"

# Load Julia
module load languages/julia/1.10.3
julia --project=. --threads=8 program.jl

echo "Job finished at $(date)"