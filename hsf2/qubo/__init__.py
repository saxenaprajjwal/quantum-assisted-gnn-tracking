"""QUBO formulations of track reconstruction sub-problems (QX3) and exploratory quantum tracking (QX4, QX5).

selection  QX3: selection of track candidates (conflict QUBO = maximum-weight independent set), per conflict
           component <= 24 variables, solved by exact enumeration, SA and QAOA (Phase-1 numerics)
triplet    Bapst-style triplet QUBO on small detector sectors (formulation of Bapst et al. 2020)
"""
