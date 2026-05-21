# Tasks: Peer Download Distribution

## Overview

Implement peer-to-peer model download distribution so nodes download models from cluster peers instead of HuggingFace when another node already has the model.

## Tasks

### 1. Add file-serving and model availability endpoints to the API

- [x] 1.1 Add `GET /api/models/{model_id:path}/available` endpoint that checks if the model is fully downloaded locally using `resolve_existing_model` and returns `{"available": true/false, "model_directory": "..."}`.
  - File: `src/exo/api/main.py`
  - Use `resolve_existing_model(model_id, card)` from `exo.download.download_utils`
  - Return the model directory path when available (needed for file serving)
  - Handle model_id path parameter (e.g., `Qwen/Qwen3.5-4B`)

- [x] 1.2 Add `GET /api/files/{model_id:path}` endpoint that lists all files available for a model (returns JSON array of relative file paths).
  - File: `src/exo/api/main.py`
  - Scan the model directory returned by `resolve_existing_model`
  - Return relative paths from the model root (e.g., `["model-00001-of-00002.safetensors", "config.json", ...]`)

- [x] 1.3 Add `GET /api/files/{model_id:path}/{filename:path}` endpoint that serves individual model files using `FileResponse`.
  - File: `src/exo/api/main.py`
  - Validate the file exists within the model directory (prevent path traversal)
  - Use FastAPI's `FileResponse` for efficient streaming
  - Set appropriate content-type headers

### 2. Modify download coordinator to prefer peer downloads

- [x] 2.1 Add a `_try_download_from_peer` method to `DownloadCoordinator` that queries other nodes for model availability and downloads from a peer if available.
  - File: `src/exo/download/coordinator.py`
  - Accept the cluster topology (node IPs) from state or configuration
  - Query each peer's `/api/models/{model_id}/available` endpoint with a short timeout (2s)
  - If a peer has the model, download all files from that peer's `/api/files/` endpoint
  - Write files to the local HuggingFace cache directory structure
  - Emit download progress events during peer download
  - Dependencies: Task 1.1, 1.2, 1.3

- [x] 2.2 Integrate peer download into `_start_download` flow — before calling `self.shard_downloader.ensure_shard()`, attempt peer download first.
  - File: `src/exo/download/coordinator.py`
  - In `_start_download_task`, before the HuggingFace download, call `_try_download_from_peer`
  - If peer download succeeds, emit `DownloadCompleted` and return
  - If peer download fails (no peers have it, network error), fall back to HuggingFace
  - Log which source was used (peer IP or HuggingFace)
  - Dependencies: Task 2.1

### 3. Add peer download configuration and node discovery

- [x] 3.1 Add node IP discovery from cluster state topology for peer querying.
  - File: `src/exo/download/coordinator.py`
  - The coordinator needs access to the cluster topology to know other node IPs
  - Add a `topology_provider` callback or pass the state's topology
  - Extract node addresses from `state.topology` (multiaddrs contain IP:port)
  - Filter out self (by `self.node_id`)
  - Use port 52415 (the API port) for peer queries
  - Dependencies: Task 2.1
