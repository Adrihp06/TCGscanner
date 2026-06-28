# Public Release Checklist

Use this checklist before pushing the public repository.

## What Goes Public

- Scanner source code, local web UI, API server, tests, and reproducible scripts.
- Small catalog metadata under `data/`.
- Documentation for architecture, setup, evaluation, and training workflow.

## What Stays Out

- Model binaries: `models/`.
- Vector DB: `data/vector_db/`.
- Training/evaluation outputs: `runs/`, `reports/`.
- Downloaded datasets: `dataset/`, `data/huggingface/`.
- Downloaded card images and user samples: `images/official/`, `images/pricecharting/`, `images/user_samples/`.
- Local certificates: `certs/`.
- Any private development branch or product roadmap implementation.
- Local planning notes under `private_planning/`.

## Pre-Push Commands

```bash
git status --short
uv run python -m unittest discover -s tests
node --check public/app.js
rg -n "hf_|TOKEN|SECRET|ACCESS|password|api[_-]?key|local-key|local-cert" .
```

The token scan can produce false positives in docs and lockfiles. Review every hit before pushing.

## Suggested Public Remote Flow

```bash
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin public-cleanup:main
```

Keep commercial app work in a private repository. Do not push private branches to the public remote.
