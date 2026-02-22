# Technical Design: fastmcp Upgrade to 3.0.1

## Architecture
- The server uses `FastMCP` framework for tool registration and life cycle management.
- Upgrade involves purely dependency management.

## Implementation Details
1. Use `uv` to update the lock file.
2. Verify that `fastmcp` 3.0.1 is picked up.

## Verification Strategy
- **Unit Tests**: Run `pytest` to catch any indirect regressions.
- **Integration**: Start server and verify tool registration.
