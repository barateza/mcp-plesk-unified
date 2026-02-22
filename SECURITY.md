# Security policy

## Report security vulnerabilities

If you discover a security vulnerability in Plesk Unified, **do not** open a
public GitHub issue. Instead, follow responsible disclosure practices.

### How to report

1. **Email**: Send a detailed report to [advisory@barateza.org](mailto:advisory@barateza.org)
2. **Telegram**: Message [@barateza](https://t.me/barateza)
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

4. **Response time**: We aim to respond within 48 hours and work on a fix
   promptly.

### What to expect

- Acknowledgment of receipt.
- Investigation timeline.
- Coordinated disclosure date.
- Credit in the security advisory (if desired).

## Security considerations

### For users

1. **Update dependencies regularly**: Keep your environment up to date
   with the latest dependency versions:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Secure knowledge base access**: The vector database may contain sensitive
   documentation.
   - Restrict server access to trusted networks.
   - Enforce appropriate file permissions on the storage directory.

3. **Verify model downloads**: First-run initialization downloads ML models
   (~1GB).
   - Use a clean environment.
   - Verify checksums if possible.
   - Only download data over HTTPS.

4. **Protect Git credentials**: When cloning private repositories:
   - Use SSH keys or personal access tokens.
   - Never commit credentials to version control.

### For developers

1. **Manage dependencies**:
   - Review dependencies before updating.
   - Check for known vulnerabilities: run `pip check`.
   - Keep the Python version current (3.12+).

2. **Validate input**:
   - Sanitize all user inputs.
   - Validate file paths.
   - Limit query sizes.

3. **Handle errors**:
   - Obfuscate sensitive paths in error messages.
   - Log securely without exposing sensitive data.
   - Use appropriate log levels.

## Supported versions

| Version | Status | Security updates |
|---------|--------|------------------|
| 0.2.x   | Active | Yes              |
| 0.1.x   | Active | Yes              |

## Security best practices

### Deployment

- Run behind a firewall or reverse proxy.
- Set configuration values using environment variables.
- Enable HTTPS if exposing the service over a network.
- Implement rate limiting.
- Monitor for unusual access patterns.

### Development

- Use a dedicated development environment.
- Never hardcode credentials.
- Review code changes carefully.
- Keep dependencies minimal.
- Use static analysis tools.

## Dependency security

This project depends on:

- **sentence-transformers**: Deep learning library for embeddings
- **lancedb**: Vector database
- **fastmcp**: MCP server framework
- **beautifulsoup4**: HTML parsing
- **gitpython**: Git operations

All dependencies come from trusted, well-maintained projects. We monitor for
security updates and keep versions current.

## Acknowledgments

Thank you for keeping Plesk Unified secure. We credit researchers who 
responsibly disclose vulnerabilities, if desired.

## Additional resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)

