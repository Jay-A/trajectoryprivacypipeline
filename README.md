# Trajectory Privacy Pipeline

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jay-a.github.io/trajectoryprivacypipeline/)

A configuration-driven framework for privacy-preserving mobility trajectory processing.  
The system is designed around a layered architecture consisting of a control plane, execution engine, and modular compute operators for spatial processing, Local Differential Privacy (LDP), and statistical evaluation.

---

## Overview

This project implements a modular pipeline for transforming raw mobility trajectories into privacy-preserving and analytically useful representations.

The core design separates concerns into three layers:

- **Control Plane**: Compiles declarative pipeline configurations (YAML/JSON) into executable DAGs using a stage registry.
- **Execution Engine**: Orchestrates runtime execution over an immutable pipeline context using view extraction and copy-on-write updates.
- **Compute Layer**: Implements modular transformation stages including preprocessing, discretization, LDP mechanisms, aggregation, and evaluation.

This structure enables reproducible experiments, flexible pipeline composition, and systematic analysis of privacyb
utility trade-offs in mobility systems.

---

## Key Features

- Config-driven pipeline specification (YAML/JSON)
- Modular stage-based execution model
- Local Differential Privacy (LDP) support for trajectory data
- Immutable context-based execution semantics
- Pluggable compute modules (preprocessing, aggregation, evaluation)
- Designed for extensibility and benchmarking

---

## Planned Repository Structure


```text
repo/
|-- src/
|   |-- control_plane/
|   |-- engine/
|   |-- compute/
|   |-- io/
|   |-- utils/
|
|-- bindings/        # C++ <-> Python interface layer (if needed)
|-- submodules/      # external dependencies (optional)
|-- configs/         # pipeline + experiment configurations
|-- docs/            # figures, system diagrams, documentation
|-- scripts/         # utilities and CLI tools
|-- tests/           # unit and integration tests
```


---

## Conceptual Model

The system models mobility trajectory processing as a structured transformation pipeline:

raw trajectories b

Each stage operates on a shared but immutable execution context.

---

## Development Status

This project is in early-stage development.  
Core focus areas include:

- Execution engine stabilization
- Compute module implementation
- Control plane compilation layer
- Evaluation and benchmarking framework

---

## External Dependencies

This project uses the Simulation of Urban MObility (SUMO) traffic simulator for microscopic mobility simulation and trajectory generation.

SUMO is an open-source, microscopic traffic simulation framework developed by the Institute of Transportation Systems at the German Aerospace Center (DLR).

If you use SUMO in research, please cite:

P. Alvarez Lopez et al., "Microscopic Traffic Simulation using SUMO", IEEE Intelligent Transportation Systems Conference (ITSC), 2018.

## License

This project is licensed under the Apache License 2.0.  
See the [LICENSE](LICENSE) file for details.
