# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Plesk Unified, please **do not** open a public GitHub issue. Instead, follow responsible disclosure practices.

### How to Report

1. **Email**: Send a detailed report to [advisory@barateza.org](mailto:advisory@barateza.org)
2. **Telegram**: Message [@barateza](https://t.me/barateza)
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

3. **Response Time**: We aim to respond within 48 hours and will work on a fix promptly

### What to Expect

- Acknowledgment of receipt
- Investigation timeline
- Coordinated disclosure date
- Credit in the security advisory (if desired)

## Security Considerations

### For Users

1. **Keep Dependencies Updated**: Regularly update to the latest versions
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Secure Knowledge Base Access**: The vector database may contain sensitive documentation
   - Restrict server access to trusted networks
   - Use appropriate file permissions on the storage directory

3. **Model Downloads**: First run downloads ML models (~1GB)
   - Use a clean environment
   - Verify checksums if possible
   - Only download over HTTPS

4. **Git Credentials**: When cloning private repositories
   - Use SSH keys or personal access tokens
   - Never commit credentials to version control

### For Developers

1. **Dependency Management**:
   - Review dependencies before updating
   - Check for known vulnerabilities: `pip check`
   - Keep Python version current (3.12+)

2. **Input Validation**:
   - Sanitize all user inputs
   - Validate file paths
   - Limit query sizes

3. **Error Handling**:
   - Don't expose sensitive paths in error messages
   - Log securely without sensitive data
   - Use appropriate log levels

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 0.1.x   | Active | Yes              |

## Security Best Practices

### Deployment

- Run behind a firewall or reverse proxy
- Use environment variables for configuration
- Enable HTTPS if exposing over network
- Implement rate limiting
- Monitor for unusual access patterns

### Development

- Use a dedicated development environment
- Never hardcode credentials
- Review code changes carefully
- Keep dependencies minimal
- Use static analysis tools

## Dependency Security

This project depends on:

- **sentence-transformers**: Deep learning library for embeddings
- **lancedb**: Vector database
- **fastmcp**: MCP server framework
- **beautifulsoup4**: HTML parsing
- **gitpython**: Git operations

All dependencies are from trusted, well-maintained projects. We monitor for security updates and aim to stay current.

## Acknowledgments

Thank you for helping keep Plesk Unified secure. Researchers who responsibly disclose vulnerabilities will be credited if desired.

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
