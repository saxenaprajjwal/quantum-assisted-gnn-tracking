"""The GNN tuning problem (T1: eight inference-time knobs) for the Phase-1 surrogate -> QUBO -> QAOA pipeline.

space      the 24-bit (8 knobs x 3 bits) search space and the 16-bit QX1 sub-grid, as Phase-1 ParamSpace/Encoding
evaluator  exact, fast evaluation of many knob settings on fixed events (shares every stage that can be shared)
workers    event-parallel evaluation in worker processes (each worker owns a fixed set of events and its caches)
runner     GNNRunner: the Phase-1 CKFRunner contract (evaluate/records/score) on the GNN pipeline, with a
           persistent, fingerprinted evaluation cache; LookupRunner: the same contract on an exhaustive landscape
phase1     builds a Phase-1 acts_qaoa Run around a Part-II runner, so the Phase-1 stages run unchanged; noise model;
           IBM account status (no QPU time)
landscape  QX1: exhaustive 16-bit landscape of the real objective
scaling    QX6: the same surrogate problem at 16-32 qubits (mixed-bit QUBO, chunked exact scan, Aer statevector QAOA)
compare    QX1 controlled comparison of optimisers on the exact landscape (many seeds, equal budgets)
baselines  grid random search and black-box simulated annealing on the real objective
"""
