#!/usr/bin/env python
"""
simulate_gpu_util.py

A simple script to simulate GPU utilization by repeatedly performing large matrix multiplications.
It logs iteration time, GPU memory usage, and a dummy loss metric to Weights & Biases.
"""

import time
import torch
import wandb


def simulate_gpu_load(num_iterations=100, matrix_size=1024):
    # Determine device: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create two random matrices on the device.
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    total_time = 0.0

    for i in range(num_iterations):
        start_time = time.time()

        # Perform a matrix multiplication to simulate heavy computation.
        # You can chain multiple operations to increase the load.
        C = A @ B
        # Apply a non-linearity
        result = torch.relu(C)
        # Force synchronization by calling .sum().item()
        dummy_loss = result.sum().item()

        iteration_time = time.time() - start_time
        total_time += iteration_time

        # Log GPU memory usage (in MB) and iteration time to wandb.
        current_memory_MB = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        wandb.log({
            "iteration": i,
            "iteration_time": iteration_time,
            "current_memory_MB": current_memory_MB,
            "dummy_loss": dummy_loss,
        })

        print(f"Iteration {i}: time {iteration_time:.4f}s, memory {current_memory_MB:.2f} MB, loss {dummy_loss:.4f}")

    avg_time = total_time / num_iterations
    print(f"Average iteration time: {avg_time:.4f}s")
    wandb.log({"avg_iteration_time": avg_time})
    return avg_time


def main():
    # Initialize wandb with your project and entity.
    wandb.init(project="PhantoMS_Retrieval", entity="jozefov-iocb-prague")
    # Optionally update configuration settings.
    wandb.config.update({
        "num_iterations": 100,
        "matrix_size": 1024,
    })

    simulate_gpu_load(num_iterations=10000, matrix_size=10024)

    wandb.finish()


if __name__ == "__main__":
    main()