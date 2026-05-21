# Implement Peer Download Distribution — Ready to Execute

## Priority: HIGH — User is actively blocked by this

## What to Build

When a node needs to download a model, check if ANY other node in the cluster already has it. If yes, download from that node's local API instead of HuggingFace.

## Implementation Plan

### Step 1: Add file-serving endpoint to the API

File: `src/exo/api/main.py`

Add `GET /api/files/{model_id:path}/{filename}` that serves files from the local HuggingFace cache directory (`~/.cache/huggingface/hub/models--{org}--{model}/snapshots/...`).

### Step 2: Add "has model" check endpoint

File: `src/exo/api/main.py`

Add `GET /api/models/{model_id:path}/available` that returns `{"available": true/false}` based on whether the model is fully downloaded locally (check download status or scan cache dir).

### Step 3: Modify the download coordinator

File: `src/exo/download/coordinator.py`

In `_start_download()`, before calling the HuggingFace downloader:
1. Query all other nodes' `/api/models/{model_id}/available` endpoint
2. If any node has it, download files from that node's `/api/files/` endpoint instead
3. If no node has it, fall back to HuggingFace download

### Key Context

- Each node runs its own API on port 52415
- Node IPs: 10.1.1.12 (gremlin-1), 10.1.1.13 (gremlin-2), 10.1.1.14 (gremlin-3), 10.1.1.15 (gremlin-4)
- Models are stored in HuggingFace cache format: `~/.cache/huggingface/hub/models--{org}--{model}/`
- The download coordinator already has `self.node_id` to identify itself
- The state has `self.state.downloads` which tracks which nodes have completed downloads
- The `DOWNLOAD_COMMANDS` topic uses `PublishPolicy.Always` (broadcast to all nodes)
- Each node's coordinator only processes commands with matching `target_node_id`

### Existing Spec Reference

The `twenty-tokens-per-second` spec (tasks 8.1-8.3) describes this feature in detail:
- `.kiro/specs/twenty-tokens-per-second/design.md` — Section "5. Model Loader Optimizer"
- `.kiro/specs/twenty-tokens-per-second/tasks.md` — Tasks 8.1, 8.2, 8.3

### Testing

After implementation:
1. SSH to gremlin-2: `ssh root@10.1.1.13`
2. Delete the model cache: `rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B`
3. Trigger a download from the dashboard
4. Check gremlin-2 logs: should show downloading from 10.1.1.12 (or another node that has it), NOT from huggingface.co
