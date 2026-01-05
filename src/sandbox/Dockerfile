# Minimal sandbox for incident-aware-agents experiment
#
# Build:
#   docker build -t incident-sandbox -f src/sandbox/Dockerfile .
#
# The container provides:
#   - Basic shell utilities
#   - Python for task execution
#   - /workspace as the only writable directory
#   - No network access (enforced at runtime)

FROM python:3.12-slim

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash sandbox

# Create workspace
RUN mkdir -p /workspace && chown sandbox:sandbox /workspace

# Create quarantine directory (for safe alternatives)
RUN mkdir -p /quarantine && chown sandbox:sandbox /quarantine

# Switch to non-root user
USER sandbox
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD echo "ok" || exit 1

# Default command
CMD ["/bin/bash"]
