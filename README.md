## SSLAP

This library provides implementations for solvers for **S**uper **S**parse **L**inear **A**ssignment **P**roblems.

An [assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) is one where a one-to-one assignment has to be made between two sets, where each assignment has an associated cost.

In super sparse assignment problems, typically less than 1% of all feasible assignments are allowed.

This library provides a Cython implementation of the [Auction Algorithm](https://en.wikipedia.org/wiki/Auction_algorithm) [1], which is well suited for super sparse problems. It is one in which *people* in one set 'bid' for *objects* in the other set, driving up their prices in order to find an optimal assignment. 

Also provided is an implementation of the [Hopcroft-Karp Algorithm](https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm) [2] for finding a maximum matching in a bipartite graph. This is used by the Auction solver to check that a given problem has a valid solution.

## Installation

```pip install sslap```

## Usage

## Benchmarking

The algorithm is best suited for large and sparse problems, where it outperforms `scipy.optimize.linear_sum_assignment`.

See below for some timed comparisons of the runtime for problems of varying density (% of valid entries in the matrix) and matrix size.

![](figs/density_benchmarking.png)
![](figs/size_benchmarking.png)

## Notes

[1] Bertsekas, D. [A Distributed Algorithm for the Assignment Problem](https://www.mit.edu/~dimitrib/Orig_Auction.pdf) (1979)
[2] Hopcroft J. Karp, R. An n^(5/2) algorithm for maximum matchings in bipartite graphs (1973)

- A matrix passed into from_matrix requires positive values only, and -1 indicates invalid values.

