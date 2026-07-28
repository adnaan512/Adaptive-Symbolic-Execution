# ML Label Construction for Symbolic Execution

To train Machine Learning models (Phase 4) to rank execution states, we need a dataset mapping `ExecutionStateFeatures (X)` to a `Target Priority Score (y)`.

Since KLEE doesn't natively score states with an "optimal" value, we must heuristically construct these labels by observing the actual coverage gains during baseline runs.

## Construction Pipeline

1. **Run Baselines:** Execute KLEE on benchmark programs using existing heuristics (`random-path`, `nurs:covnew`, etc.) and log the execution state features at regular intervals.
2. **Track Coverage Horizons:** For a given state $S_i$ at time $t$, observe the total branch coverage achieved by KLEE at time $t + \Delta t$ (the *horizon*). 
3. **Compute Gain:** 
   $$y_i = Coverage(t + \Delta t) - Coverage(t)$$
4. **Discounting (Optional):** Apply a temporal discount factor so that states leading to *immediate* coverage gains are scored higher than states leading to gains much later.
5. **Normalization:** Normalize the $y$ targets to $[0, 1]$ or use standard scaling for regression stability.

## Why Regression?

We formulate this as a **Regression** problem rather than binary Classification. Predicting a continuous coverage gain provides a granular priority queue.

For Neural Networks (`NeuralNetRanker`), predicting a continuous score via Mean Squared Error (MSE) loss stabilizes gradient updates compared to sparse binary targets.

## Integration with Phase 4

The `models.ml.trainer.ModelTrainer` class expects `labels: list[float]`. The offline data pipeline (to be executed in Phase 7 for dataset generation) will compute these $y$ values and pair them with the `ExecutionStateFeatures` generated in Phase 3.
