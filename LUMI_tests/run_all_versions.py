#!/usr/bin/env python
"""
run_all_versions.py

A driver script that runs simulate_gpu_util_with_gnn.py for multiple torch module versions.
It uses the LUMI module system to load each version before invoking the simulation.
Each run logs its results to wandb under a run name that includes the torch module version.
"""

import subprocess

# List of torch module versions available on LUMI for ROCm.
torch_versions = ["2.5", "2.4", "2.2", "2.1", "2.0"]

# Number of iterations and graph parameters for testing
num_iterations = 1000
num_nodes = 5000
num_edges = 20000
num_features = 32

for version in torch_versions:
    print(f"\n=== Running simulation for torch version: {version} ===")
    # Build the bash command:
    #   1. Use the CSC modulefiles directory.
    #   2. Load the desired pytorch module.
    #   3. Run the simulation script with the given parameters.
    cmd = (
        f"bash -c 'module use /appl/local/csc/modulefiles/; "
        f"module load pytorch/{version}; "
        f"python LUMI/simulate_gpu_util_with_gnn.py "
        f"--torch_version {version} "
        f"--num_iterations {num_iterations} "
        f"--num_nodes {num_nodes} "
        f"--num_edges {num_edges} "
        f"--num_features {num_features}'"
    )
    print("Running command:")
    print(cmd)
    # Run the command; if any run fails, raise an error.
    subprocess.run(cmd, shell=True, check=True)