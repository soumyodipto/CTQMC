# Continuous-time Quantum Monte Carlo (CTQMC)

This module contains a CTQMC algorithm in C++ and Python used to evaluate thermal equilibrium observables of quantum particles in continuous space at ultra-low temperatures where nuclear quantum effects are significant.

The mathematical statement of the problem is evaluating the sum of an infinite series where each term in the series is an integral of dimension given by the index of the term. For example, the first term is a 1d integral, the second a 2-d, and so on. The series is converging but we need to evaluate really high order terms for that!
