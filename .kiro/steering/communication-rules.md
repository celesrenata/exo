# Communication Rules

## Language

- Never use hedge words: `may`, `might`, `probably`, `possibly`, `perhaps`, `likely`, `could be`
- Never use minimizing words: `easy`, `easiest`, `simple`, `simplify`, `just`, `trivial`, `straightforward`
- These words signal uncertainty or dismissiveness. If you don't know something, say "I don't know" and investigate. If something took hours to get right, respect that effort in how you describe it.
- State facts. Ask questions. Don't hedge.

## NixOS Debugging

- If a command-line tool is not available in the current environment, use `nix-shell -p <package>` to get it temporarily.
- Example: `nix-shell -p jq --run 'curl -s http://localhost:52415/state | jq .topology.nodes | jq length'`
- Do not assume tools are unavailable — use `nix-shell -p` to make them available.
