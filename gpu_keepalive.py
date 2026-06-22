"""GPU keep-alive: maintain GPU utilization above a target threshold.

Uses a file-based lock so the training benchmark gets ZERO interference:
  - Training creates /tmp/gpu_bench.lock before benchmarking
  - Keep-alive detects the lock and fully pauses
  - Training removes the lock after benchmarking

Usage:
    nohup python gpu_keepalive.py > /tmp/keepalive.log 2>&1 &
    kill $(cat /tmp/gpu_keepalive.pid)
"""

import os
import signal
import sys
import time

import torch

LOCK_FILE = "/tmp/gpu_bench.lock"
PID_FILE = "/tmp/gpu_keepalive.pid"


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    def cleanup(signum, frame):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    device = torch.device("cuda")
    N = 1024
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    print(f"[keepalive] pid={os.getpid()}, matrix={N}x{N} (~{N*N*4*3/1e6:.0f}MB)", flush=True)
    print(f"[keepalive] strategy: continuous compute with lock-based pause", flush=True)

    tick = 0
    while True:
        # If benchmark is running, fully stop and wait
        if os.path.exists(LOCK_FILE):
            time.sleep(0.3)
            continue

        # Continuous small bursts: 100 matmuls then yield briefly
        # 100 × ~0.1ms = ~10ms compute, then 5ms yield → ~67% duty cycle
        for _ in range(100):
            torch.mm(a, b)
        torch.cuda.synchronize()
        time.sleep(0.005)

        tick += 1
        if tick % 5000 == 1:
            print(f"[keepalive] alive, tick={tick}", flush=True)


if __name__ == "__main__":
    main()
