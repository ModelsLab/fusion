# Fusion

Fusion is a cross-platform Go CLI for model and kernel optimization workflows. The long-term goal is one CLI that can plan GPU-specific optimizations, generate Triton or CUDA kernel candidates, run them on local or remote Linux GPU machines, benchmark before vs after, and keep the winning variants.

Today Fusion already gives you a useful foundation:

- ModelsLab-backed chat sessions and Modelslab-only auth
- browser-based `fusion login` that hands off from modelslab.com back into the local CLI
- an embedded optimization knowledge base with GPU profiles, strategies, skills, examples, and source references
- a public Markdown-first `knowledgebase/` corpus that compiles into the shipped SQLite index
- a packed SQLite BM25 search index generated from the curated knowledge files
- host capability detection with explicit warnings on unsupported setups
- target management for `local`, `ssh`, and `sim`
- benchmark and profile execution against those targets
- persisted artifacts for before/after comparisons
- target-aware optimization planning
- optimization sessions that persist retrieved context, backend candidates, and stage artifacts
- CuTe DSL, Triton, and CUDA workspace scaffolding with build and verify flows

Fusion can run on macOS for planning, artifact management, ModelsLab setup, and SSH orchestration. Real CUDA compilation, profiling, and authoritative kernel performance validation still need a Linux machine with NVIDIA tooling.

## Status

What works today:

- `fusion` and `fusion chat` as a chat-first agent entry point
- `fusion env detect|doctor`
- `fusion generate keychain`
- `fusion optimize plan` with a curated GPU and optimization knowledge base
- `fusion kb list|search|show|context` backed by an embedded SQLite BM25 index
- `fusion update kb` to rebuild a local Markdown-backed knowledge snapshot and SQLite index
- `fusion optimize session create|list|show`
- `fusion optimize cute init|build|verify|benchmark`
- `fusion optimize triton init|build|verify|benchmark`
- `fusion optimize cuda init|build|verify|benchmark`
- `fusion target add|list|show|remove|default`
- `fusion target exec` and `fusion target copy`
- `fusion benchmark run` and `fusion benchmark compare`
- `fusion profile run`
- release packaging for Linux, macOS, and Windows

What is not implemented yet:

- Modelslab-backed Triton/CUDA/CuTe code generation
- automatic optimization loops that generate, run, score, and retain winning kernels

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

## Quick Start

Connect Fusion to ModelsLab in the browser:

```bash
fusion login
```

Or configure it manually for CI or headless environments:

```bash
fusion auth set \
  --token "$MODELSLAB_API_KEY" \
  --model openai-gpt-5.4-pro
```

Store Hugging Face and GitHub tokens for model and private-repo workflows:

```bash
fusion hf login --token "$HF_TOKEN"
fusion github login --token "$GITHUB_TOKEN"
```

Validate them:

```bash
fusion hf whoami
fusion github whoami
```

Fusion shell commands automatically expose:

- `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
- `GITHUB_TOKEN`, `GH_TOKEN`

That lets the agent download models from Hugging Face and work against private GitHub repos. For private HTTPS git operations, prefer `gh` commands or `git` with an Authorization header using `$GITHUB_TOKEN` instead of embedding secrets into URLs.

Start the interactive agent shell:

```bash
fusion
```

Run a single natural-language turn:

```bash
fusion chat "optimize qwen2.5-72b for 4090 decode latency and compare AWQ vs Triton"
```

Inside chat, Fusion can use tools for:

- listing, reading, writing, replacing, and deleting files
- running shell commands locally or on configured targets
- searching the knowledge base
- creating optimization sessions and retrieving skill/context packets
- building optimization plans
- running benchmark and profile workflows
- scaffolding and running CuTe DSL, Triton, and CUDA workspaces

Chat-local commands:

```text
/help
/tools
/session
/exit
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
fusion kb context --gpu b200 --workload decode --operators attention,kv-cache --precision fp8 --runtime vllm
```

Rebuild a private local knowledge base from Markdown docs:

```bash
fusion update kb
```

This bootstraps `~/.config/fusion/knowledgebase/` if needed, rebuilds the SQLite index under `~/.config/fusion/knowledge/`, and makes future Fusion runs prefer that rebuilt local knowledge base.

See what the current machine can and cannot do:

```bash
fusion env detect
fusion env doctor --backend all --fix-script
```

Register a remote Ubuntu target over SSH:

```bash
fusion generate keychain --name gpulab
```

Paste the printed public key into your GPU provider, then register the target with the generated private key path:

```bash
fusion target add \
  --name lab-4090 \
  --mode ssh \
  --host 203.0.113.10 \
  --user ubuntu \
  --gpu rtx4090 \
  --key ~/.ssh/id_ed25519 \
  --remote-dir ~/fusion \
  --default
```

Register a non-authoritative proxy/sim target:

```bash
fusion target add \
  --name sim-h100-on-4090 \
  --mode sim \
  --gpu h100 \
  --proxy-gpu rtx4090
```

List configured targets:

```bash
fusion target list
fusion target show --name lab-4090
```

Run a command directly on a target:

```bash
fusion target exec --name lab-4090 --command "nvidia-smi"
```

Copy files to a remote target:

```bash
fusion target copy \
  --name lab-4090 \
  --src ./kernels \
  --dst ~/fusion/kernels \
  --recursive
```

Plan optimizations for a configured target:

```bash
fusion optimize plan \
  --target lab-4090 \
  --model llama-3.1-8b \
  --workload decode \
  --operator attention \
  --operator kv-cache \
  --precision bf16
```

Create a CuTe DSL workspace and compile or verify it on a target:

```bash
fusion optimize cute init \
  --name cute-add-one \
  --output ./cute-add-one \
  --gpu-arch sm90

fusion optimize cute build \
  --workspace ./cute-add-one \
  --target lab-4090 \
  --gpu-arch sm89

fusion optimize cute verify \
  --workspace ./cute-add-one \
  --target lab-4090 \
  --gpu-arch sm89

fusion optimize cute benchmark \
  --workspace ./cute-add-one \
  --target lab-4090 \
  --gpu-arch sm89
```

Create a session-backed Triton or CUDA candidate loop:

```bash
fusion optimize session create \
  --name qwen-b200 \
  --gpu b200 \
  --model qwen2.5-72b \
  --workload decode \
  --operator attention \
  --operator kv-cache \
  --precision fp8 \
  --runtime vllm \
  --query "optimize qwen decode attention on b200"

fusion optimize triton init \
  --session <session-id> \
  --name attention-triton

fusion optimize cuda init \
  --session <session-id> \
  --name attention-cuda

fusion optimize triton build \
  --session <session-id> \
  --candidate triton-attention-triton

fusion optimize triton verify \
  --session <session-id> \
  --candidate triton-attention-triton

fusion optimize session show --id <session-id>
```

The same session flow now works for CuTe candidates:

```bash
fusion optimize cute init \
  --session <session-id> \
  --name attention-cute

fusion optimize cute benchmark \
  --session <session-id> \
  --candidate cute-dsl-attention-cute
```

Or plan for a GPU directly:

```bash
fusion optimize plan \
  --gpu rtx4090 \
  --model llama-3.1-8b \
  --workload decode \
  --operator attention \
  --operator kv-cache \
  --precision bf16
```

Run a benchmark and compare before/after artifacts:

```bash
fusion benchmark run \
  --target lab-4090 \
  --name before \
  --command "python benchmark.py"

fusion benchmark run \
  --target lab-4090 \
  --name after \
  --command "python benchmark_optimized.py"

fusion benchmark compare \
  --before ~/Library/Application\\ Support/fusion/artifacts/benchmarks/<before>.json \
  --after ~/Library/Application\\ Support/fusion/artifacts/benchmarks/<after>.json
```

Pass metrics explicitly when your benchmark command does not print them:

```bash
fusion benchmark run \
  --target lab-4090 \
  --name before \
  --command "python benchmark.py >/tmp/bench.log" \
  --metrics "tokens_per_sec=142.5 latency_ms=7.9"
```

Run a profile command on a remote or local target:

```bash
fusion profile run \
  --target lab-4090 \
  --tool ncu \
  --command "ncu --set full python benchmark.py"
```

## Target Modes

Fusion supports three execution modes:

- `local`: run on the current machine
- `ssh`: run on a remote Linux machine over SSH
- `sim`: use a proxy machine or proxy GPU while targeting another GPU profile

Recommended usage:

- use `local` when the current machine actually has the intended NVIDIA stack
- use `ssh` for real Ubuntu GPU boxes
- use `sim` for rough iteration, compatibility work, and non-authoritative proxy runs

`sim` mode is intentionally explicit about its limitations. It does not emulate an H100, B200, or any other GPU with performance fidelity on top of a different GPU. It is useful for:

- iterating against a target GPU profile
- validating command and artifact flow
- rough proxy benchmarking with warnings

Authoritative performance numbers still require the real target GPU.

## Host Limitations

Fusion reports host limitations with `fusion env detect`.

On macOS, expect:

- planning and artifact workflows to work
- SSH orchestration to work
- local CUDA compilation to be unavailable unless the host actually has a supported NVIDIA stack
- local Nsight profiling to be unavailable in normal modern macOS setups

In practice, macOS is best treated as a control plane for:

- planning
- ModelsLab login and session setup
- target registration
- remote execution over SSH
- comparing benchmark and profile artifacts

## Benchmark And Profile Artifacts

Fusion stores artifacts under the user config directory. On macOS this is typically:

```bash
~/Library/Application Support/fusion/artifacts
```

Current artifact types:

- `benchmarks/*.json`
- `profiles/*.json`

Benchmark metrics are parsed from:

- JSON printed to stdout, for example `{"tokens_per_sec": 125, "latency_ms": 8}`
- key/value lines, for example `tokens_per_sec=125`
- the optional `--metrics` flag

`fusion benchmark compare` compares wall time plus any common metric keys found in both artifacts.

## Command Summary

Core commands:

- `fusion login`
- `fusion auth login|show|set|logout`
- `fusion env detect`
- `fusion gpu detect|normalize`
- `fusion kb list|search|show|context`
- `fusion optimize plan`
- `fusion target add|list|show|remove|default|exec|copy`
- `fusion benchmark run|compare`
- `fusion profile run`

## Testing

Run the full test suite:

```bash
go test ./...
```

Current tests cover:

- knowledge-base loading and search
- optimization planner scoring
- artifact metric parsing
- target validation
- target resolution
- local and sim execution behavior
- local file and directory copy behavior
- benchmark comparison helper logic

Run formatters before opening a PR:

```bash
gofmt -w $(find . -name '*.go' -print)
```

## Repository Layout

- `cmd/fusion`: CLI entrypoint
- `internal/config`: local config and ModelsLab token storage
- `internal/modelslab`: ModelsLab API and browser-login constants
- `internal/system`: host and toolchain detection
- `internal/targets`: target validation and execution semantics
- `internal/runner`: local and SSH command/copy execution
- `internal/artifacts`: benchmark and profile artifact storage
- `internal/kb`: embedded knowledge base loader, SQLite BM25 search, and context packet compiler
- `internal/optimize`: optimization planner and recommendation engine
- `knowledge`: source-backed GPU, strategy, skill, example, and search-index assets embedded into the binary
- `scripts`: install helpers and knowledge-index generation
- `.github/workflows`: CI and tagged release pipelines
- `.goreleaser.yaml`: cross-platform packaging config
- `docs`: architecture and roadmap notes

## Current Gaps

Fusion is not yet a full autonomous kernel writer. The missing pieces are important:

- Modelslab-backed Triton/CUDA generation
- kernel correctness verification
- Triton/CUDA compile pipelines
- session-oriented optimization loops
- promotion logic for winning kernels per GPU family and workload shape

Those pieces should build on the current target, benchmark, profile, and artifact foundation instead of bypassing it.

## Roadmap

1. Modelslab-backed Triton/CUDA kernel generation inside the CLI
2. correctness verification commands for generated kernels
3. first-class compile commands for Triton and CUDA C++
4. structured optimization sessions that chain plan, generate, benchmark, profile, and compare
