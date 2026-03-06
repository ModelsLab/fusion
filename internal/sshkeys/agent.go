package sshkeys

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func AddToAgent(privatePath string) (string, error) {
	if strings.TrimSpace(privatePath) == "" {
		return "", fmt.Errorf("private key path is required")
	}
	if strings.TrimSpace(os.Getenv("SSH_AUTH_SOCK")) == "" {
		return "ssh-agent not detected; skipped agent add", nil
	}

	args := []string{privatePath}
	if runtime.GOOS == "darwin" {
		args = []string{"--apple-use-keychain", privatePath}
	}

	cmd := exec.Command("ssh-add", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if runtime.GOOS == "darwin" {
			fallback := exec.Command("ssh-add", privatePath)
			stderr.Reset()
			fallback.Stderr = &stderr
			if fallback.Run() == nil {
				return "added private key to ssh-agent", nil
			}
		}
		message := strings.TrimSpace(stderr.String())
		if message == "" {
			message = err.Error()
		}
		return "", fmt.Errorf("ssh-add failed: %s", message)
	}

	if runtime.GOOS == "darwin" {
		return "added private key to ssh-agent and Apple keychain", nil
	}
	return "added private key to ssh-agent", nil
}
