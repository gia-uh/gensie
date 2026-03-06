# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-06

### Added
- **Core SDK**: `GenSIEAgent` and `Participant` abstractions for building extraction agents.
- **OpenAI Baseline**: `BasicAgent` reference implementation using Structured Outputs.
- **Evaluation Engine**: Standardized "Flattened Schema Scoring" metrics (Precision, Recall, Micro-F1).
- **FastAPI Server**: Standardized `/run` and `/info` endpoints for competition serving.
- **Unified CLI**: `gensie serve` and `gensie eval` commands for local development and benchmarking.
- **Dockerization**: Optimized `Dockerfile` and `docker-compose.yml` with volume support and `uv` integration.
- **Starter Data**: 40 silver-generated and manually curated instances covering Legal, Medical, Cultural, and Technical domains.
- **Documentation**: Comprehensive guides for the starter kit and submission process.
