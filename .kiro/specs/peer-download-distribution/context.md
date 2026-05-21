# Feature: Peer-to-Peer Model Download Distribution

## Problem

When a model is placed across multiple nodes, each node downloads the full model independently from HuggingFace. This wastes bandwidth (4x the download for a 4-node cluster) and is slow.

## Desired Behavior

1. ONE node downloads the model from HuggingFace
2. Other nodes download from the first node via the cluster network (10.1.1.0/24, 1Gbps+)
3. The cluster network is faster than the internet connection, so this saves time and bandwidth

## Current Architecture

- `src/exo/download/coordinator.py` — DownloadCoordinator runs on each node
- `src/exo/download/impl_shard_downloader.py` — Downloads from HuggingFace Hub
- `DOWNLOAD_COMMANDS` topic (PublishPolicy.Always) — broadcast to all nodes
- Master sends `StartDownload` command per node via placement logic
- Each node's coordinator processes only commands with matching `target_node_id`
- Downloads go to `~/.cache/huggingface/hub/` on each node

## Proposed Solution

1. Add a file-serving HTTP endpoint on each node: `GET /api/files/<model_id>/<filename>`
2. When a node receives a `StartDownload` command, check if another node already has the model (via download status in state)
3. If another node has it, download from that node's file-serving endpoint instead of HuggingFace
4. If no node has it, download from HuggingFace (first node to start wins)
5. Use the existing download progress events to track which nodes have completed downloads

## Key Files

- `src/exo/download/coordinator.py` — Main download logic
- `src/exo/download/impl_shard_downloader.py` — HuggingFace download implementation
- `src/exo/master/placement.py` — Where download commands are generated
- `src/exo/shared/types/commands.py` — StartDownload command
- `src/exo/shared/types/state.py` — State.downloads tracks per-node download status

## Cluster Info

- 4 nodes on 10.1.1.0/24 subnet
- Each node has ~94GB RAM, shared memory architecture
- Models stored in HuggingFace cache format
- Qwen3.5-4B is ~9GB, larger models are 20-50GB+
