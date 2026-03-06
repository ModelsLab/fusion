package knowledge

import "embed"

// Files bundles the curated optimization knowledge so the CLI can run without
// a live network dependency.
//
//go:embed knowledge.db
var Files embed.FS
