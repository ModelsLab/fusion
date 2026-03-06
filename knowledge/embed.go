package knowledge

import "embed"

// Files bundles the curated optimization knowledge so the CLI can run without
// a live network dependency.
//
//go:embed *.json
var Files embed.FS
