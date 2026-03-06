package sshkeys

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateEd25519CreatesKeypair(t *testing.T) {
	dir := t.TempDir()
	key, err := GenerateEd25519(GenerateRequest{
		Name:   "GPU Lab",
		Output: filepath.Join(dir, "fusion-test-key"),
	})
	if err != nil {
		t.Fatalf("GenerateEd25519() error = %v", err)
	}

	if _, err := os.Stat(key.PrivatePath); err != nil {
		t.Fatalf("expected private key to exist: %v", err)
	}
	if _, err := os.Stat(key.PublicPath); err != nil {
		t.Fatalf("expected public key to exist: %v", err)
	}
	if !strings.HasPrefix(key.PublicKey, "ssh-ed25519 ") {
		t.Fatalf("expected ssh-ed25519 public key, got %q", key.PublicKey)
	}
}

func TestGenerateEd25519RefusesOverwriteByDefault(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(dir, "fusion-test-key")
	if _, err := GenerateEd25519(GenerateRequest{Name: "first", Output: output}); err != nil {
		t.Fatalf("first GenerateEd25519() error = %v", err)
	}
	if _, err := GenerateEd25519(GenerateRequest{Name: "second", Output: output}); err == nil {
		t.Fatal("expected overwrite protection error")
	}
}
