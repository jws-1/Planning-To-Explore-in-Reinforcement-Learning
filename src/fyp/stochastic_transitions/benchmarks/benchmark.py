from .frozen_lake import BenchmarkFrozenLake
from .windy_gridworld import BenchmarkWindyGridworld
from .gridworld import BenchmarkGridworld

if __name__ == "__main__":


    # windy_gridworld = BenchmarkWindyGridworld()
    # windy_gridworld.perform_benchmarks(m=2, n=100, p=10, w = 5)

    frozen_lake = BenchmarkFrozenLake()
    frozen_lake.perform_benchmarks(m=20, n=100, p=10, w=1)

    # gridworld = BenchmarkGridworld()
    # gridworld.perform_benchmarks(m=10, n=100, p=10, w=5)