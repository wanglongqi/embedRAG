# API Reference

This page provides the detailed API documentation for EmbedRAG, automatically generated from the docstrings in the source code.

## CLI

The command-line interface for managing and running EmbedRAG nodes.

::: embedrag.cli

## Configuration

EmbedRAG's configuration system, including writer and query node settings.

::: embedrag.config

## Models

Data structures used for communication and internal storage.

### API Models
Request and response models for the FastAPI endpoints.
::: embedrag.models.api

### Chunking Models
Internal models for representing document chunks.
::: embedrag.models.chunk

### Manifest
Models for the snapshot manifest that coordinates index loading.
::: embedrag.models.manifest

## Shared Utilities

Core utilities shared across the project.

### Object Store
Abstraction layer for S3, TOS, and MinIO storage.
::: embedrag.shared.object_store

### Metrics
Prometheus metrics collection and reporting.
::: embedrag.shared.metrics

## Writer Node

Components specific to the writer node, which handles ingestion and indexing.

### Ingestion & Build
The writer's FastAPI application and lifespan management.
::: embedrag.writer.app

### Index Builder
The logic for constructing FAISS indexes from document vectors.
::: embedrag.writer.index_builder

### Storage
SQLite-based storage for document text and metadata.
::: embedrag.writer.storage

## Query Node

Components specific to the query node, which handles search and retrieval.

### Search & Retrieval
The query node's FastAPI application and retrieval logic.
::: embedrag.query.app

### Dense Retrieval
FAISS-based vector search implementation.
::: embedrag.query.retrieval.dense

### Sparse Retrieval
SQLite FTS5-based keyword search implementation.
::: embedrag.query.retrieval.sparse

### Fusion
Reciprocal Rank Fusion (RRF) for combining dense and sparse results.
::: embedrag.query.retrieval.fusion

## Clustering

The standalone, reusable clustering module that powers the `embedrag cluster`
CLI and the integrated query-node cluster API.

### Pipeline & Library API
Orchestration and the public `cluster_vectors` / `cluster_items` entry points.
::: embedrag.cluster.pipeline

### Sources
Loading vectors and items from files, `.npy`, a writer DB, or TF-IDF.
::: embedrag.cluster.source

### Algorithms
Pluggable clustering backends behind a single interface.
::: embedrag.cluster.algorithms

### Evaluation
Internal/external metrics and the automatic parameter sweep.
::: embedrag.cluster.evaluate

### Explainability
c-TF-IDF keywords, medoids, cohesion/separation, and attribution.
::: embedrag.cluster.explain

### Persistence
Side-file storage for cluster runs.
::: embedrag.cluster.store
