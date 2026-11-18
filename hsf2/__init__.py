"""HSF Part II: GNN track reconstruction (Interaction Network) and quantum-assisted tuning of it.

Sub-packages
    data      events (TrackML, ODD export, synthetic detector) as one EventData type
    graph     loose "module-map-lite" graph construction and inference-time edge filters
    model     edge-classifying Interaction Network (pure PyTorch), training, quantum edge head
    tracking  track building from edge scores, matching and metrics
    tuning    the GNN-tuning problem for the Phase-1 surrogate -> QUBO -> QAOA pipeline
    qubo      QUBO track selection, triplet QUBO, solvers, linear-system (HHL / 1-bit filter) tracking

Command line:  python -m hsf2 <command>   (python -m hsf2 --help)
"""
__version__ = "0.2.21"
