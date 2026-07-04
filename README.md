# snl-progress-core-rs

A high-performance Rust extension designed to accelerate the computational bottlenecks within the [snl-progress](https://github.com/sandialabs/snl-progress) Python solver framework.

By leveraging PyO3 and Maturin, this module seamlessly replaces heavy Python-side data preparation and dispatch loops—specifically targeting functions like `RAUtilities.OptDispatchMP`—with pre-compiled, memory-safe Rust binaries.

## Architecture & Goals
* **Targeted Acceleration:** Executes multi-period data manipulation at near C-level speeds prior to Pyomo ingestion, drastically reducing the overhead of running 10,000+ stochastic simulation loops.
* **Seamless Integration:** Acts as a drop-in Python module (`snl_progress_core_rs`), bypassing Python's execution limitations for computationally dense hot paths while leaving the core Pyomo algebraic modeling intact.
* **Cloud & HPC Optimization:** Minimizes vCPU-second consumption when fanned out across distributed orchestration environments (e.g., Azure Container Instances), enabling cost-effective parallel processing.

## Installation (End-Users)

For standard usage alongside the `snl-progress` solver, you do not need Rust installed. Simply install the pre-compiled wheel via PyPI:

```bash
pip install snl-progress-core-rs