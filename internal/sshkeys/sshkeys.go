package sshkeys

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

type GenerateRequest struct {
	Name      string
	Comment   string
	Output    string
	Overwrite bool
}

type GeneratedKey struct {
	Name        string
	Comment     string
	PrivatePath string
	PublicPath  string
	PublicKey   string
}

func GenerateEd25519(req GenerateRequest) (*GeneratedKey, error) {
	name := sanitizeName(req.Name)
	if name == "" {
		name = "default"
	}

	privatePath, err := resolvePrivateKeyPath(name, req.Output)
	if err != nil {
		return nil, err
	}
	publicPath := privatePath + ".pub"

	if !req.Overwrite {
		if _, err := os.Stat(privatePath); err == nil {
			return nil, fmt.Errorf("private key already exists at %s; pass --overwrite to replace it", privatePath)
		}
		if _, err := os.Stat(publicPath); err == nil {
			return nil, fmt.Errorf("public key already exists at %s; pass --overwrite to replace it", publicPath)
		}
	}

	if err := os.MkdirAll(filepath.Dir(privatePath), 0o700); err != nil {
		return nil, fmt.Errorf("create ssh key dir: %w", err)
	}

	comment := strings.TrimSpace(req.Comment)
	if comment == "" {
		comment = defaultComment(name)
	}

	if req.Overwrite {
		_ = os.Remove(privatePath)
		_ = os.Remove(publicPath)
	}

	cmd := exec.Command("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", comment, "-f", privatePath)
	if output, err := cmd.CombinedOutput(); err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
		}
		return nil, fmt.Errorf("ssh-keygen failed: %s", message)
	}

	if err := os.Chmod(privatePath, 0o600); err != nil {
		return nil, fmt.Errorf("chmod private key: %w", err)
	}
	if err := os.Chmod(publicPath, 0o644); err != nil {
		return nil, fmt.Errorf("chmod public key: %w", err)
	}

	publicBytes, err := os.ReadFile(publicPath)
	if err != nil {
		return nil, fmt.Errorf("read public key: %w", err)
	}
	authorized := strings.TrimSpace(string(publicBytes))

	return &GeneratedKey{
		Name:        name,
		Comment:     comment,
		PrivatePath: privatePath,
		PublicPath:  publicPath,
		PublicKey:   strings.TrimSpace(authorized),
	}, nil
}

func resolvePrivateKeyPath(name, output string) (string, error) {
	output = strings.TrimSpace(output)
	if output != "" {
		return filepath.Abs(output)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home dir: %w", err)
	}
	return filepath.Join(homeDir, ".ssh", "fusion-"+name+"-ed25519"), nil
}

func defaultComment(name string) string {
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		return "fusion-" + name
	}
	return "fusion-" + name + "@" + strings.TrimSpace(host)
}

var nameSanitizer = regexp.MustCompile(`[^a-zA-Z0-9._-]+`)

func sanitizeName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	if name == "" {
		return ""
	}
	name = nameSanitizer.ReplaceAllString(name, "-")
	name = strings.Trim(name, "-.")
	return name
}
