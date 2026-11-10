# Security policy

euler-known-datasets extracts archives downloaded from the internet, so path
handling during extraction is security-relevant: no archive member may be
written outside the output directory, and links, devices and FIFOs are never
extracted.

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue.

Use [GitHub's private vulnerability reporting
form](https://github.com/d-rothen/euler-known-datasets/security/advisories/new)
and include the affected version, impact, reproduction steps, and any suggested
mitigation. Remove credentials and private dataset contents from the report
unless they are strictly necessary to reproduce the issue.
