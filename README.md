# Distributed Parallel Matrix-Vector Multiplication using OpenMPI

## Overview
This project implements a distributed-memory parallel **Matrix-Vector Multiplication** ($y = A \times x$, where $N = 4000$) across two separate networked Linux systems (**System 1** and **System 2**) using **OpenMPI (C)** over TCP/IP.

Unlike shared-memory multithreading (such as OpenMP), each system operates in its own isolated memory space and exchanges data explicitly over a network interface using MPI collective communication primitives (`MPI_Bcast`, `MPI_Scatter`, and `MPI_Gather`).

---

## System & Network Architecture

The cluster consists of two containerized Ubuntu 24.04 nodes connected via a dedicated virtual Ethernet bridge network (`172.28.0.0/24`):

```text
+-----------------------------------+              +-----------------------------------+
|      System 1 (Master Node)       |              |      System 2 (Worker Node)       |
|      Hostname: system1            |              |      Hostname: system2            |
|      IP Address: 172.28.0.10      |              |      IP Address: 172.28.0.11      |
|      MPI Rank: 0                  |              |      MPI Rank: 1                  |
+-----------------+-----------------+              +-----------------+-----------------+
                  |                                                  |
                  | === 1. Passwordless SSH Process Launch =========>|
                  | === 2. MPI_Bcast (Vector x: 32 KB) =============>|
                  | === 3. MPI_Scatter (2000 Matrix Rows: ~61 MB) ==>|
                  |                                                  |
                  |         [ Parallel CPU Computation ]             |
                  |                                                  |
                  | <== 4. MPI_Gather (2000 Result Elements: 16 KB) =|
                  |                                                  |
                  +------- [ Bridge Network: 172.28.0.0/24 ] --------+
```

* **System 1 (`172.28.0.10`, Rank 0 – Master):** Allocates the $4000 \times 4000$ double-precision matrix ($122.07\text{ MB}$) and input vector $x$, broadcasts $x$ and scatters half of the matrix rows to System 2 over TCP/IP, computes the first $2000$ rows locally, gathers the partial output vector from System 2, and verifies numerical correctness against a sequential CPU baseline.
* **System 2 (`172.28.0.11`, Rank 1 – Worker):** Launches remotely via passwordless SSH, starts with empty data buffers, receives its $2000$-row partition over the `172.28.0.0/24` network interface, computes partial dot products in parallel, and transmits the resulting $2000$-element slice back to System 1.

---

## Project Structure

```text
mpi-distributed-matvec/
├── Dockerfile              # Defines Ubuntu 24.04 nodes with GCC, OpenMPI, and OpenSSH Server
├── docker-compose.yml      # Provisions System 1 (172.28.0.10) & System 2 (172.28.0.11) on mpi_net
├── hostfile                # Maps networked IP addresses and CPU slots for OpenMPI
├── mpi_matvec.c            # Distributed C implementation using MPI_Bcast, MPI_Scatter, MPI_Gather
└── README.md               # Setup, execution, and testing documentation
```

---

## Prerequisites

* **Docker & Docker Compose** (runs locally on Linux/macOS/Windows or in the cloud via **GitHub Codespaces** with zero local installation)
* **Software inside nodes (automatically installed via `Dockerfile`):**
  * `build-essential` (`gcc`)
  * `openmpi-bin` and `libopenmpi-dev` (`mpicc`, `mpirun`)
  * `openssh-server` and `iputils-ping`

---

## Step-by-Step Setup & Execution

### 1. Build and Start the Two-Node Networked Cluster
Build the Ubuntu MPI image and start both `system1` and `system2` on the `172.28.0.0/24` subnet:

```bash
docker compose up -d --build
```

Verify that both containers are active:

```bash
docker ps
```

### 2. Test Network Connectivity & Passwordless SSH
Confirm ICMP ping connectivity and passwordless SSH authentication from **System 1** (`172.28.0.10`) to **System 2** (`172.28.0.11`):

```bash
docker exec system1 ping -c 2 172.28.0.11
docker exec system1 ssh root@172.28.0.11 "hostname && hostname -I"
```

### 3. Compile the MPI Source Code
Compile `mpi_matvec.c` with `-O3` optimization inside the shared `/workspace` directory and verify the binary is accessible on System 2:

```bash
docker exec system1 mpicc -O3 /workspace/mpi_matvec.c -o /workspace/mpi_matvec
docker exec system2 ls -lh /workspace/mpi_matvec
```

### 4. Execute the Distributed Program Across Both Systems
Run `mpirun` across both IP addresses defined in `hostfile`, explicitly restricting MPI's TCP Byte Transfer Layer (`btl_tcp_if_include`) to the `172.28.0.0/24` network bridge:

```bash
docker exec system1 mpirun --allow-run-as-root \
  --hostfile /workspace/hostfile \
  --mca btl_tcp_if_include 172.28.0.0/24 \
  -np 2 /workspace/mpi_matvec
```

---

## Testing & Verification Results

### Verification Methodology
1. **Network Transport Enforcement:** Passing `--mca btl_tcp_if_include 172.28.0.0/24` ensures all inter-node transfers (`MPI_Bcast`, `MPI_Scatter`, and `MPI_Gather`) traverse the TCP/IP network stack between `172.28.0.10` and `172.28.0.11`.
2. **Numerical Accuracy Check:** Upon gathering all $4000$ output elements, Rank 0 computes a sequential reference vector $y_{\text{ref}} = A \times x$ and calculates the maximum absolute error $\max_i |y_{\text{dist}}[i] - y_{\text{ref}}[i]|$.

### Actual Execution Output

```text
==================================================================
 Distributed Parallel Matrix-Vector Multiplication (4000 x 4000)
==================================================================
[System 1 | Host: system1 | Rank 0] Dataset allocated (122.07 MB).
[Host: system1 | Rank 0] Received 2000 matrix rows over network in 0.1729 s. Computing...
[Host: system2 | Rank 1] Received 2000 matrix rows over network in 0.1729 s. Computing...

------------------------------------------------------------------
[System 1 | Host: system1 | Rank 0] Execution Complete!
  -> Total Distributed Time (Scatter + Compute + Gather): 0.1849 s
  -> Sample Output Vector Elements: y[0]=11200.00, y[2000]=11200.00, y[3999]=10800.00
  -> Max Verification Error vs Sequential CPU: 0.00e+00 (PASS)
==================================================================
```

---

## Cleanup

To stop and remove the two virtual nodes and network bridge after testing:

```bash
docker compose down
```