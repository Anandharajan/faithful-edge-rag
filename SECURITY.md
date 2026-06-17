# Security Policy

## Supported Versions

This project is pre-1.0. Security fixes are applied to the main branch until release branches are defined.

## Reporting a Vulnerability

Open a private security advisory if the repository is hosted on GitHub. If advisories are not enabled, contact the maintainers through the repository owner.

Do not include private datasets, credentials, raw edge payloads, or exploitable details in public issues.

## Data Handling

- Do not commit secrets.
- Do not commit raw private documents or telemetry.
- Prefer synthetic datasets for tests.
- Use policy checks before moving edge data to central services.

