# Fusion

Fusion is a cross-platform Go CLI for GPU inference optimization planning. It is designed to run on macOS for design and planning, then execute profiling and kernel work on Linux machines with NVIDIA GPUs.

Current scope:

- manage provider tokens for OpenAI, Anthropic, OpenRouter, Gemini, and Groq
- ship an embedded optimization knowledge base with GPUs, strategies, and references
- detect the local host toolchain and visible NVIDIA hardware
- build ranked optimization plans for a target GPU, workload, precision, and operator mix

This first pass intentionally focuses on planning and knowledge curation because the current host has no CUDA stack. The next phase is Linux execution: compile, verify, benchmark, and profile kernels remotely or locally on Ubuntu.

## Install

### Prebuilt binaries

Linux and macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/ModelsLab/fusion/main/scripts/install.sh | sh
```

Pin a specific release or install into a custom directory:

```bash
curl -fsSL https://raw.githubusercontent.com/ModelsLab/fusion/main/scripts/install.sh | \
  FUSION_VERSION=v0.1.0 INSTALL_DIR="$HOME/.local/bin" sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/ModelsLab/fusion/main/scripts/install.ps1 | iex
```

### From source

```bash
go install github.com/ModelsLab/fusion/cmd/fusion@latest
```

### Local build

```bash
make build
./bin/fusion version
```

### GitHub release workflow

Push a version tag and GitHub Actions will publish `tar.gz` and `.zip` assets for:

- Linux `amd64`, `arm64`
- macOS `amd64`, `arm64`
- Windows `amd64`, `arm64`

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow uploads matching archives plus `checksums.txt`.

## Examples

Store a provider token:

```bash
fusion auth set --provider openai --token "$OPENAI_API_KEY" --default
```

Inspect the current host:

```bash
fusion env detect
fusion gpu detect
```

Search the embedded optimization corpus:

```bash
fusion kb search "paged attention"
fusion kb show --kind gpu --id rtx4090
```

Plan optimizations for a remote Linux target:

```bash
fusion optimize plan \
  --gpu rtx4090 \
  --model llama-3.1-8b \
  --workload decode \
  --operator attention \
  --operator kv-cache \
  --precision bf16
```

## Repository Layout

- `cmd/fusion`: CLI entrypoint
- `internal/config`: local config and token storage
- `internal/providers`: provider registry for OpenAI-compatible model backends
- `internal/system`: host and toolchain detection
- `internal/kb`: embedded knowledge base loader and search
- `internal/optimize`: optimization planner and recommendation engine
- `knowledge`: source-backed GPU and optimization data embedded into the binary
- `scripts`: install helpers for release binaries
- `.github/workflows`: CI and tagged release pipelines
- `.goreleaser.yaml`: cross-platform packaging config
- `docs`: architecture and roadmap notes

## What Comes Next

1. Linux execution engine for compile, verify, benchmark, and profile loops
2. remote runner for Ubuntu GPU boxes over SSH or cloud APIs
3. provider-backed agent flows that synthesize Triton/CUDA candidates from the curated knowledge base
4. structured benchmark artifacts so Fusion can compare before/after runs and keep winning kernels
