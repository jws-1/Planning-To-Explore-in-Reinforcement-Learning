from .cliff_walking import BenchmarkCliffWalking
from .windy_gridworld import BenchmarkWindyGridworld
from .mountain_car import BenchmarkMountainCar
from .gridworld import BenchmarkGridworld

if __name__ == "__main__":
    cliff_walking = BenchmarkCliffWalking()
    cliff_walking.perform_benchmarks(m=10, n=100, p=10, w = 5)

    windy_gridworld = BenchmarkWindyGridworld()
    windy_gridworld.perform_benchmarks(m=10, n=100, p=10, w = 5)

    # # # mountain_car = BenchmarkMountainCar()
    # # # mountain_car.perform_benchmarks(m=1, n=1000, p=30, w = 10)
    gridworld = BenchmarkGridworld()
    gridworld.perform_benchmarks(m=10, n=100, p=10, w = 5)