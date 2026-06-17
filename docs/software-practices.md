# Software Practices

## Engineering Standards

- Keep core experiments reproducible with Docker Compose and pinned model/config metadata.
- Use typed Python and strict schema validation at service boundaries.
- Keep retrieval, generation, orchestration, evaluation, and governance as separate modules.
- Track prompt versions, model checkpoints, embedding models, vector index parameters, and dataset versions.
- Prefer deterministic evaluation runs with recorded random seeds.

## Quality Gates

- `ruff check .`
- `mypy src`
- `pytest`
- Container build validation before release tags.
- Evaluation regression runs before claiming research improvements.

## Security Practices

- Do not commit secrets, datasets with private content, or model weights.
- Use `.env.example` for configuration shape only.
- Sign edge batches.
- Enforce local-only and summary-only policies before sync.
- Log access and tool calls without leaking private content.

## GitHub Workflow

- Use pull requests for all changes.
- Require CI to pass before merge.
- Use issue templates for bugs and research tasks.
- Record research claims in issues or PR descriptions with dataset, model, hardware, and metric references.

