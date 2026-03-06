package runner

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func localCopy(src, dst string, recursive bool) error {
	info, err := os.Stat(src)
	if err != nil {
		return fmt.Errorf("stat source: %w", err)
	}

	if info.IsDir() {
		if !recursive {
			return fmt.Errorf("source %s is a directory; pass --recursive", src)
		}
		return copyDir(src, dst)
	}

	return copyFile(src, dst)
}

func copyDir(src, dst string) error {
	return filepath.WalkDir(src, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		relative, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(dst, relative)

		if entry.IsDir() {
			return os.MkdirAll(targetPath, 0o755)
		}

		return copyFile(path, targetPath)
	})
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("create destination dir: %w", err)
	}

	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open source: %w", err)
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("create destination: %w", err)
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return fmt.Errorf("copy contents: %w", err)
	}

	info, err := os.Stat(src)
	if err == nil {
		_ = os.Chmod(dst, info.Mode())
	}

	return nil
}

func scpCopy(ctx context.Context, req CopyRequest) error {
	if strings.TrimSpace(req.Target.Host) == "" {
		return fmt.Errorf("ssh copy requires a host")
	}

	destination := req.Target.Host
	if req.Target.User != "" {
		destination = req.Target.User + "@" + destination
	}

	args := []string{}
	if req.Recursive {
		args = append(args, "-r")
	}
	if req.Target.Port > 0 {
		args = append(args, "-P", fmt.Sprintf("%d", req.Target.Port))
	}
	if strings.TrimSpace(req.Target.IdentityFile) != "" {
		args = append(args, "-i", req.Target.IdentityFile)
	}

	args = append(args, req.Source, destination+":"+req.Dest)
	cmd := exec.CommandContext(ctx, "scp", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("scp copy: %w: %s", err, strings.TrimSpace(string(output)))
	}

	return nil
}
