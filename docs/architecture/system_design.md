# System Design

The project is organised into modular components:

* **Models** – located under `src/models/` and include decision trees, transformers and ensembles.
* **Features** – feature engineering transformers in `src/features/`.
* **Strategies** – trading logic in `src/strategies/`.
* **Scripts** – helper scripts and experiments.

Data flows from raw CSV files through preprocessing into model training and strategy execution.
