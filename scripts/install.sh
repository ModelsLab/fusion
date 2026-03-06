#!/usr/bin/env sh

set -eu

REPO="${FUSION_REPO:-ModelsLab/fusion}"
VERSION="${FUSION_VERSION:-}"
INSTALL_DIR="${INSTALL_DIR:-}"

log() {
  printf '%s\n' "$*" >&2
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

detect_os() {
  case "$(uname -s)" in
    Darwin) printf 'darwin' ;;
    Linux) printf 'linux' ;;
    *)
      log "unsupported OS: $(uname -s)"
      exit 1
      ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'amd64' ;;
    arm64|aarch64) printf 'arm64' ;;
    *)
      log "unsupported architecture: $(uname -m)"
      exit 1
      ;;
  esac
}

resolve_version() {
  if [ -n "$VERSION" ]; then
    printf '%s' "$VERSION"
    return
  fi

  latest_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")"
  version="$(printf '%s' "$latest_url" | awk -F/ '{print $NF}')"
  if [ -z "$version" ]; then
    log "failed to resolve latest release from GitHub"
    exit 1
  fi
  printf '%s' "$version"
}

verify_checksum() {
  asset="$1"
  if command -v shasum >/dev/null 2>&1; then
    (cd "$tmpdir" && grep " ${asset}\$" checksums.txt | shasum -a 256 -c - >/dev/null)
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$tmpdir" && grep " ${asset}\$" checksums.txt | sha256sum -c - >/dev/null)
    return
  fi

  log "warning: skipping checksum verification because neither shasum nor sha256sum is available"
}

choose_install_dir() {
  if [ -n "$INSTALL_DIR" ]; then
    printf '%s' "$INSTALL_DIR"
    return
  fi

  if [ -w /usr/local/bin ]; then
    printf '%s' /usr/local/bin
    return
  fi

  printf '%s' "${HOME}/.local/bin"
}

need_cmd curl
need_cmd tar
need_cmd awk
need_cmd grep

os="$(detect_os)"
arch="$(detect_arch)"
version="$(resolve_version)"
version_nov="${version#v}"
asset="fusion_${version_nov}_${os}_${arch}.tar.gz"
base_url="https://github.com/${REPO}/releases/download/${version}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT INT TERM

log "downloading ${asset} from ${base_url}"
curl -fsSL "${base_url}/${asset}" -o "${tmpdir}/${asset}"
curl -fsSL "${base_url}/checksums.txt" -o "${tmpdir}/checksums.txt"
verify_checksum "$asset"

tar -xzf "${tmpdir}/${asset}" -C "$tmpdir"
destination="$(choose_install_dir)"
mkdir -p "$destination"

if command -v install >/dev/null 2>&1; then
  install "${tmpdir}/fusion" "${destination}/fusion"
else
  cp "${tmpdir}/fusion" "${destination}/fusion"
  chmod +x "${destination}/fusion"
fi

log "installed fusion to ${destination}/fusion"
case ":${PATH}:" in
  *":${destination}:"*) ;;
  *)
    log "add ${destination} to PATH if it is not already exported"
    ;;
esac
