# test_gpu_compute.py
import torch
import time
import wandb


def main():
    print("PyTorch version:", torch.__version__)
    gpu_count = torch.cuda.device_count()
    print("Number of GPUs available:", gpu_count)

    # Initialize wandb (the WANDB_API_KEY should already be in the environment)
    run = wandb.init(project="test_lumi_gpu_compute", reinit=True)

    # Do some dummy GPU work: multiply two random matrices repeatedly for ~10 seconds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)

    iterations = 0
    start_time = time.time()
    duration = 10  # seconds
    while time.time() - start_time < duration:
        c = torch.matmul(a, b)
        if iterations % 10 == 0:
            torch.cuda.synchronize()  # force sync to measure time accurately
        iterations += 1

    total_time = time.time() - start_time
    print(f"Completed {iterations} iterations in {total_time:.2f} seconds.")

    # Log the number of iterations to wandb
    wandb.log({"iterations": iterations, "duration_sec": total_time})
    run.finish()


if __name__ == "__main__":
    main()