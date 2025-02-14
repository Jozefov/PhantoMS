# #!/usr/bin/env python
# """
# simulate_gpu_util_with_gnn.py
#
# A simple script to simulate GPU utilization by performing GNN computations on dummy graphs.
# Logs iteration time, GPU memory usage, and a dummy loss metric to Weights & Biases.
# """
#
# import time
# import torch
# import torch_geometric
# import wandb
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
#
#
# class SimpleGNN(torch.nn.Module):
#     """ A simple two-layer GCN model """
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(SimpleGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels
#     def forward(self, x, edge_index):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
#
#
# def create_dummy_graph(num_nodes=1000, num_edges=5000, num_features=16):
#     """ Creates a random graph with the specified number of nodes and edges """
#     x = torch.randn((num_nodes, num_features))  # Node features
#     edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
#     y = torch.randn((num_nodes, 1))  # Dummy target labels
#
#     return Data(x=x, edge_index=edge_index, y=y)
#
#
# def simulate_gnn_load(num_iterations=100, num_nodes=1000, num_edges=5000, num_features=16):
#     """ Simulates GPU load using a simple GNN on a random graph """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Generate dummy graph and move to device
#     graph = create_dummy_graph(num_nodes, num_edges, num_features).to(device)
#
#     # Define GNN model and optimizer
#     model = SimpleGNN(in_channels=num_features, hidden_channels=32, out_channels=1).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_fn = torch.nn.MSELoss()
#
#     total_time = 0.0
#
#     for i in range(num_iterations):
#         start_time = time.time()
#
#         # Forward pass
#         optimizer.zero_grad()
#         out = model(graph.x, graph.edge_index)
#         loss = loss_fn(out, graph.y)
#
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#
#         iteration_time = time.time() - start_time
#         total_time += iteration_time
#
#         # Log GPU memory usage (in MB) and iteration time to wandb
#         current_memory_MB = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
#         wandb.log({
#             "iteration": i,
#             "iteration_time": iteration_time,
#             "current_memory_MB": current_memory_MB,
#             "dummy_loss": loss.item(),
#         })
#
#         print(f"Iteration {i}: time {iteration_time:.4f}s, memory {current_memory_MB:.2f} MB, loss {loss.item():.4f}")
#
#     avg_time = total_time / num_iterations
#     print(f"Average iteration time: {avg_time:.4f}s")
#     wandb.log({"avg_iteration_time": avg_time})
#     return avg_time
#
#
# def main():
#     # Initialize wandb
#     wandb.init(project="PhantoMS_Retrieval", entity="jozefov-iocb-prague")
#     wandb.config.update({
#         "num_iterations": 100,
#         "num_nodes": 1000,
#         "num_edges": 5000,
#         "num_features": 16,
#     })
#
#     simulate_gnn_load(num_iterations=100000, num_nodes=5000, num_edges=20000, num_features=32)
#
#     wandb.finish()
#
#
# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python
# """
# simulate_gpu_util_with_gnn.py
#
# A simple script to simulate GPU utilization by performing GNN computations on dummy graphs.
# Logs iteration time, GPU memory usage, and a dummy loss metric to Weights & Biases.
# """
#
# import time
# import torch
# import torch_geometric
# import wandb
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
#
#
# class SimpleGNN(torch.nn.Module):
#     """ A simple two-layer GCN model """
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(SimpleGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
#
#
# def create_dummy_graph(num_nodes=1000, num_edges=5000, num_features=16):
#     """ Creates a random graph with the specified number of nodes and edges """
#     x = torch.randn((num_nodes, num_features))  # Node features
#     edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
#     y = torch.randn((num_nodes, 1))  # Dummy target labels
#
#     return Data(x=x, edge_index=edge_index, y=y)
#
#
# def simulate_gnn_load(device, num_iterations=25000, num_nodes=1000, num_edges=5000, num_features=16):
#     """ Simulates GPU/CPU load using a simple GNN on a random graph """
#     print(f"Running simulation on {device}...")
#
#     # Generate dummy graph and move to the selected device
#     graph = create_dummy_graph(num_nodes, num_edges, num_features).to(device)
#
#     # Define GNN model and optimizer
#     model = SimpleGNN(in_channels=num_features, hidden_channels=32, out_channels=1).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_fn = torch.nn.MSELoss()
#
#     total_time = 0.0
#     start_total_time = time.time()  # Start measuring total execution time
#
#     for i in range(num_iterations):
#         start_time = time.time()
#
#         # Forward pass
#         optimizer.zero_grad()
#         out = model(graph.x, graph.edge_index)
#         loss = loss_fn(out, graph.y)
#
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#
#         iteration_time = time.time() - start_time
#         total_time += iteration_time
#
#         # Log GPU memory usage (in MB) and iteration time to wandb
#         current_memory_MB = (
#             torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
#         )
#         wandb.log({
#             f"{device}_iteration": i,
#             f"{device}_iteration_time": iteration_time,
#             f"{device}_current_memory_MB": current_memory_MB,
#             f"{device}_dummy_loss": loss.item(),
#         })
#
#         # print(f"[{device}] Iteration {i}: time {iteration_time:.4f}s, memory {current_memory_MB:.2f} MB, loss {loss.item():.4f}")
#
#     total_execution_time = time.time() - start_total_time  # Calculate total time
#     avg_time = total_time / num_iterations  # Calculate average time per iteration
#
#     print(f"[{device}] Average iteration time: {avg_time:.4f}s")
#     print(f"[{device}] Total execution time: {total_execution_time:.2f}s")
#
#     # Log average time and total execution time
#     wandb.log({
#         f"{device}_avg_iteration_time": avg_time,
#         f"{device}_total_execution_time": total_execution_time,
#     })
#
#     return avg_time, total_execution_time
#
#
# def main():
#     # Initialize wandb
#     wandb.init(project="PhantoMS_Retrieval", entity="jozefov-iocb-prague")
#     num_of_iterations = 200000
#     wandb.config.update({
#         "num_iterations": num_of_iterations,
#         "num_nodes": 5000,
#         "num_edges": 20000,
#         "num_features": 32,
#     })
#
#     # Run on CPU
#     cpu_avg_time, cpu_total_time = simulate_gnn_load(device=torch.device("cpu"), num_iterations=num_of_iterations)
#
#     # Run on GPU if available
#     if torch.cuda.is_available():
#         gpu_avg_time, gpu_total_time = simulate_gnn_load(device=torch.device("cuda"), num_iterations=num_of_iterations)
#     else:
#         gpu_avg_time, gpu_total_time = None, None
#         print("CUDA is not available. Skipping GPU test.")
#
#     # Log CPU vs GPU performance
#     wandb.log({
#         "cpu_avg_iteration_time": cpu_avg_time,
#         "cpu_total_execution_time": cpu_total_time,
#         "gpu_avg_iteration_time": gpu_avg_time if gpu_avg_time is not None else "N/A",
#         "gpu_total_execution_time": gpu_total_time if gpu_total_time is not None else "N/A",
#     })
#
#     print({
#         "cpu_avg_iteration_time": cpu_avg_time,
#         "cpu_total_execution_time": cpu_total_time,
#         "gpu_avg_iteration_time": gpu_avg_time if gpu_avg_time is not None else "N/A",
#         "gpu_total_execution_time": gpu_total_time if gpu_total_time is not None else "N/A",
#     })
#     wandb.finish()
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
"""
simulate_gpu_util_with_gnn.py

A script to simulate GPU utilization by performing GNN computations on dummy graphs.
Uses mini-batches for efficient GPU processing. Logs execution time to Weights & Biases.
"""

import time
import torch
import torch_geometric
import wandb
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader


class SimpleGNN(torch.nn.Module):
    """ A simple two-layer GCN model """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def create_dummy_graph(num_nodes=1000, num_edges=5000, num_features=16):
    """ Creates a random graph with the specified number of nodes and edges """
    x = torch.randn((num_nodes, num_features))  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
    y = torch.randn((num_nodes, 1))  # Dummy target labels

    return Data(x=x, edge_index=edge_index, y=y)


def create_dataset(num_graphs=10000, num_nodes=1000, num_edges=5000, num_features=16):
    """ Generates a dataset of random graphs """
    return [create_dummy_graph(num_nodes, num_edges, num_features) for _ in range(num_graphs)]


def simulate_gnn_load(device, num_iterations=200000, batch_size=512, num_graphs=10000):
    """ Simulates GPU/CPU load using mini-batched GNN training """
    print(f"Running simulation on {device} with batch size {batch_size}...")

    # Create dataset and DataLoader for batching
    dataset = create_dataset(num_graphs=num_graphs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define GNN model and optimizer
    model = SimpleGNN(in_channels=dataset[0].num_node_features, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    total_time = 0.0
    start_total_time = time.time()  # Start measuring total execution time

    for i in range(num_iterations):
        start_time = time.time()

        # Load a batch of graphs
        try:
            batch = next(iter(dataloader))  # Get a batch
        except StopIteration:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Reload data
            batch = next(iter(dataloader))

        batch = batch.to(device)  # Move batch to GPU or CPU

        # Forward pass
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        iteration_time = time.time() - start_time
        total_time += iteration_time

        # Log GPU memory usage (in MB) and iteration time to wandb
        current_memory_MB = (
            torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )
        wandb.log({
            f"{device}_iteration": i,
            f"{device}_iteration_time": iteration_time,
            f"{device}_current_memory_MB": current_memory_MB,
            f"{device}_dummy_loss": loss.item(),
        })

        # Print every 1000 iterations
        if i % 100 == 0:
            print(f"[{device}] Iteration {i}: time {iteration_time:.4f}s, memory {current_memory_MB:.2f} MB, loss {loss.item():.4f}")

    total_execution_time = time.time() - start_total_time  # Calculate total time
    avg_time = total_time / num_iterations  # Calculate average time per iteration

    print(f"[{device}] Average iteration time: {avg_time:.4f}s")
    print(f"[{device}] Total execution time: {total_execution_time:.2f}s")

    # Log average time and total execution time
    wandb.log({
        f"{device}_avg_iteration_time": avg_time,
        f"{device}_total_execution_time": total_execution_time,
    })

    return avg_time, total_execution_time


def main():
    # Initialize wandb
    wandb.init(project="PhantoMS_Retrieval", entity="jozefov-iocb-prague")

    num_of_iterations = 25000
    batch_size = 32  # Adjust batch size to better leverage GPU parallelism
    num_graphs = 10000  # The number of graphs in the dataset

    wandb.config.update({
        "num_iterations": num_of_iterations,
        "batch_size": batch_size,
        "num_graphs": num_graphs,
        "num_nodes": 5000,
        "num_edges": 20000,
        "num_features": 32,
    })

    # Run on GPU if available
    if torch.cuda.is_available():
        gpu_avg_time, gpu_total_time = simulate_gnn_load(device=torch.device("cuda"), num_iterations=num_of_iterations, batch_size=batch_size, num_graphs=num_graphs)
    else:
        gpu_avg_time, gpu_total_time = None, None
        print("CUDA is not available. Skipping GPU test.")

    # Run on CPU
    cpu_avg_time, cpu_total_time = simulate_gnn_load(device=torch.device("cpu"), num_iterations=num_of_iterations, batch_size=batch_size, num_graphs=num_graphs)


    # Log CPU vs GPU performance
    wandb.log({
        "cpu_avg_iteration_time": cpu_avg_time,
        "cpu_total_execution_time": cpu_total_time,
        "gpu_avg_iteration_time": gpu_avg_time if gpu_avg_time is not None else "N/A",
        "gpu_total_execution_time": gpu_total_time if gpu_total_time is not None else "N/A",
    })

    print({
        "cpu_avg_iteration_time": cpu_avg_time,
        "cpu_total_execution_time": cpu_total_time,
        "gpu_avg_iteration_time": gpu_avg_time if gpu_avg_time is not None else "N/A",
        "gpu_total_execution_time": gpu_total_time if gpu_total_time is not None else "N/A",
    })
    wandb.finish()


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """
# simulate_gpu_util_with_gnn.py
#
# A simple script to simulate GPU utilization by performing GNN computations on a dummy graph.
# It logs iteration time, GPU memory usage, and a dummy loss metric to Weights & Biases.
# The script accepts a command-line argument to record the torch module version.
# """
#
# import argparse
# import time
# import torch
# import wandb
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
#
# # A simple two-layer GCN model
# class SimpleGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(SimpleGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         x = torch.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
#
# def create_dummy_graph(num_nodes=1000, num_edges=5000, num_features=16):
#     """Creates a random graph with the specified parameters."""
#     x = torch.randn((num_nodes, num_features))  # Node features
#     edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
#     y = torch.randn((num_nodes, 1))  # Dummy target labels
#     return Data(x=x, edge_index=edge_index, y=y)
#
# def simulate_gnn_load(num_iterations=100, num_nodes=1000, num_edges=5000, num_features=16):
#     """Simulates GPU load using a simple GNN on a random graph."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Generate dummy graph and move it to the device
#     graph = create_dummy_graph(num_nodes, num_edges, num_features).to(device)
#
#     # Create the GNN model and optimizer
#     model = SimpleGNN(in_channels=num_features, hidden_channels=32, out_channels=1).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_fn = torch.nn.MSELoss()
#
#     total_time = 0.0
#
#     for i in range(num_iterations):
#         start_time = time.time()
#
#         optimizer.zero_grad()
#         out = model(graph.x, graph.edge_index)
#         loss = loss_fn(out, graph.y)
#         loss.backward()
#         optimizer.step()
#
#         iteration_time = time.time() - start_time
#         total_time += iteration_time
#
#         current_memory_MB = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
#         wandb.log({
#             "iteration": i,
#             "iteration_time": iteration_time,
#             "current_memory_MB": current_memory_MB,
#             "dummy_loss": loss.item(),
#         })
#         print(f"Iteration {i}: time {iteration_time:.4f}s, memory {current_memory_MB:.2f} MB, loss {loss.item():.4f}")
#
#     avg_time = total_time / num_iterations
#     print(f"Average iteration time: {avg_time:.4f}s")
#     wandb.log({"avg_iteration_time": avg_time})
#     return avg_time
#
# def main():
#     parser = argparse.ArgumentParser(description="Simulate GNN load and log metrics to wandb.")
#     parser.add_argument("--torch_version", type=str, default="unknown", help="Torch module version (for run naming)")
#     parser.add_argument("--num_iterations", type=int, default=50000, help="Number of iterations")
#     parser.add_argument("--num_nodes", type=int, default=1000, help="Number of nodes in dummy graph")
#     parser.add_argument("--num_edges", type=int, default=5000, help="Number of edges in dummy graph")
#     parser.add_argument("--num_features", type=int, default=16, help="Number of features per node")
#     args = parser.parse_args()
#
#     # Initialize wandb; the run name includes the torch version
#     wandb.init(project="PhantoMS_Retrieval", entity="jozefov-iocb-prague",
#                config=vars(args), name=f"torch_{args.torch_version}")
#     simulate_gnn_load(num_iterations=args.num_iterations,
#                       num_nodes=args.num_nodes,
#                       num_edges=args.num_edges,
#                       num_features=args.num_features)
#     wandb.finish()
#
# if __name__ == "__main__":
#     main()