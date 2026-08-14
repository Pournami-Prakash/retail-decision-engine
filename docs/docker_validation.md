# Docker Runtime Validation

Validated locally on 20 July 2026 using Docker Desktop.

## Evidence

- Image: `retail-decision-engine:0.2.0`
- Image digest: `sha256:827662135ca9bc535b322e2b26a4437bd93cca52c301ea5869e7a0045b9cea70`
- Image size: 287,678,205 bytes
- Build: all 10 Dockerfile stages completed successfully
- Runtime state: running and healthy
- Health response: service healthy; decision policy blocked until release gates pass
- Unmounted-artifact decision test: HTTP 422 with `causal_gate_artifact_missing`

The 422 response is expected. `.dockerignore` deliberately excludes generated
artifacts and proprietary/raw data from the image. A production deployment would
mount or retrieve a registered, read-only artifact bundle after verifying hashes.

## What this validates

This validates image construction, dependency installation, non-root execution,
port exposure, the Docker health check, HTTP routing, structured audit output,
and fail-closed behavior. It does not validate a cloud deployment, live service
levels, or the historical promotion policy.
