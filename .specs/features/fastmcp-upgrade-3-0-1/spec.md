# Feature Spec: upgrade fastmcp to 3.0.1

## Goals
- Upgrade `fastmcp` to version `3.0.1` to resolve potential bugs in middleware state and OpenAPI circular references.
- Ensure the server remains functional and stable.

## Requirements
- `fastmcp` version must be exactly `3.0.1` in `uv.lock`.
- Server must start without errors.
- Existing functionality (`refresh_knowledge`, `search_plesk_unified`) must work.

## Impact Analysis
- **Low Risk**: This is a patch release.
- **Breaking Changes**: None identified for the current usage in `plesk_unified/server.py`.
