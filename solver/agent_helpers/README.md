# agent_helpers

`jq` stream renderers that turn a coding-agent CLI's JSON event stream into
readable, timestamped **trajectory** text.

- `codex_stream.sh` — for `codex exec --json` output.
- `claude_stream.sh` — for `claude -p --output-format stream-json` output.

Used by `solver/engine/cli_agent.py` to render each candidate's
`trajectory.jsonl` → `trajectory.txt` (best-effort; skipped if `jq` is absent).
Vendored from the monorepo prototype. Require `jq` + `bash`.
