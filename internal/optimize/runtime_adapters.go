package optimize

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type RuntimeDetection struct {
	Adapter        string   `json:"adapter"`
	Matched        bool     `json:"matched"`
	Confidence     float64  `json:"confidence"`
	Reasons        []string `json:"reasons,omitempty"`
	SuggestedPaths []string `json:"suggested_paths,omitempty"`
}

type RuntimePatchOperation struct {
	Path       string `json:"path"`
	Content    string `json:"content"`
	CreateOnly bool   `json:"create_only,omitempty"`
}

type RuntimePatchPlan struct {
	Version     int                     `json:"version"`
	Adapter     string                  `json:"adapter"`
	ProjectRoot string                  `json:"project_root,omitempty"`
	Notes       []string                `json:"notes,omitempty"`
	Operations  []RuntimePatchOperation `json:"operations"`
}

type RuntimePatchRecord struct {
	Path         string `json:"path"`
	BackupPath   string `json:"backup_path,omitempty"`
	Created      bool   `json:"created"`
	OriginalHash string `json:"original_hash,omitempty"`
	AppliedHash  string `json:"applied_hash,omitempty"`
}

type RuntimePatchState struct {
	Version     int                  `json:"version"`
	ID          string               `json:"id"`
	Adapter     string               `json:"adapter"`
	ProjectRoot string               `json:"project_root"`
	Notes       []string             `json:"notes,omitempty"`
	Records     []RuntimePatchRecord `json:"records,omitempty"`
	AppliedAt   time.Time            `json:"applied_at"`
	RevertedAt  time.Time            `json:"reverted_at,omitempty"`
}

var adapterSignatures = map[string][]string{
	"transformers":   {"transformers", "AutoModel", "AutoTokenizer", "pipeline("},
	"diffusers":      {"diffusers", "DiffusionPipeline", "UNet2DConditionModel", "Transformer2DModel"},
	"vllm":           {"vllm", "LLM(", "SamplingParams"},
	"sglang":         {"sglang", "sgl."},
	"generic-python": {".py", "pyproject.toml", "requirements.txt"},
}

func DetectRuntimeAdapters(projectRoot string) ([]RuntimeDetection, error) {
	projectRoot = strings.TrimSpace(projectRoot)
	if projectRoot == "" {
		return nil, fmt.Errorf("project root is required")
	}
	files, err := collectProjectFiles(projectRoot)
	if err != nil {
		return nil, err
	}
	out := []RuntimeDetection{}
	for adapter, signatures := range adapterSignatures {
		detection := RuntimeDetection{Adapter: adapter}
		score := 0.0
		codeMatch := false
		for _, file := range files {
			lowerPath := strings.ToLower(file.Path)
			for _, signature := range signatures {
				if strings.Contains(lowerPath, strings.ToLower(signature)) || strings.Contains(strings.ToLower(file.Content), strings.ToLower(signature)) {
					weight := 0.25
					if file.Markdown {
						weight = 0.05
					} else {
						codeMatch = true
					}
					score += weight
					detection.Reasons = append(detection.Reasons, "matched signature "+signature+" in "+file.Path)
					detection.SuggestedPaths = append(detection.SuggestedPaths, file.Path)
				}
			}
		}
		if score > 1 {
			score = 1
		}
		detection.Confidence = score
		detection.SuggestedPaths = dedupeStrings(detection.SuggestedPaths)
		if adapter == "generic-python" {
			detection.Matched = codeMatch
		} else {
			detection.Matched = codeMatch
			if detection.Matched && detection.Confidence < 0.35 {
				detection.Confidence = 0.35
			}
		}
		out = append(out, detection)
	}
	return out, nil
}

func ApplyRuntimePatch(plan RuntimePatchPlan) (RuntimePatchState, string, error) {
	projectRoot := strings.TrimSpace(plan.ProjectRoot)
	if projectRoot == "" {
		return RuntimePatchState{}, "", fmt.Errorf("project root is required")
	}
	if len(plan.Operations) == 0 {
		return RuntimePatchState{}, "", fmt.Errorf("patch operations are required")
	}
	id := time.Now().UTC().Format("20060102-150405") + "-" + safeSessionPathID(plan.Adapter)
	root := filepath.Join(projectRoot, ".fusion", "runtime-patches", id)
	backupRoot := filepath.Join(root, "backups")
	if err := os.MkdirAll(backupRoot, 0o755); err != nil {
		return RuntimePatchState{}, "", fmt.Errorf("create runtime patch backup root: %w", err)
	}

	state := RuntimePatchState{
		Version:     1,
		ID:          id,
		Adapter:     strings.TrimSpace(plan.Adapter),
		ProjectRoot: projectRoot,
		Notes:       append([]string{}, plan.Notes...),
		AppliedAt:   time.Now().UTC(),
	}

	for _, op := range plan.Operations {
		targetPath, err := safeProjectPath(projectRoot, op.Path)
		if err != nil {
			return RuntimePatchState{}, "", err
		}
		record := RuntimePatchRecord{
			Path: targetPath,
		}
		original, readErr := os.ReadFile(targetPath)
		if readErr == nil {
			record.OriginalHash = contentHash(original)
			rel, err := filepath.Rel(projectRoot, targetPath)
			if err != nil {
				return RuntimePatchState{}, "", fmt.Errorf("resolve runtime patch relative path: %w", err)
			}
			backupPath := filepath.Join(backupRoot, rel)
			if err := os.MkdirAll(filepath.Dir(backupPath), 0o755); err != nil {
				return RuntimePatchState{}, "", fmt.Errorf("create runtime patch backup dir: %w", err)
			}
			if err := os.WriteFile(backupPath, original, 0o600); err != nil {
				return RuntimePatchState{}, "", fmt.Errorf("write runtime patch backup: %w", err)
			}
			record.BackupPath = backupPath
		} else if os.IsNotExist(readErr) {
			record.Created = true
		} else {
			return RuntimePatchState{}, "", fmt.Errorf("read runtime patch target: %w", readErr)
		}
		if op.CreateOnly && !record.Created {
			return RuntimePatchState{}, "", fmt.Errorf("runtime patch create-only path already exists: %s", targetPath)
		}
		if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
			return RuntimePatchState{}, "", fmt.Errorf("create runtime patch target dir: %w", err)
		}
		content := []byte(op.Content)
		record.AppliedHash = contentHash(content)
		if err := os.WriteFile(targetPath, content, 0o600); err != nil {
			return RuntimePatchState{}, "", fmt.Errorf("write runtime patch target: %w", err)
		}
		state.Records = append(state.Records, record)
	}

	statePath := filepath.Join(root, "state.json")
	if err := writeJSONFile(statePath, state); err != nil {
		return RuntimePatchState{}, "", err
	}
	return state, statePath, nil
}

func RevertRuntimePatch(statePath string) (RuntimePatchState, error) {
	var state RuntimePatchState
	if err := readJSONFile(statePath, &state); err != nil {
		return RuntimePatchState{}, err
	}
	for i := len(state.Records) - 1; i >= 0; i-- {
		record := state.Records[i]
		if record.Created && record.BackupPath == "" {
			if err := os.Remove(record.Path); err != nil && !os.IsNotExist(err) {
				return RuntimePatchState{}, fmt.Errorf("remove runtime patch created file: %w", err)
			}
			continue
		}
		original, err := os.ReadFile(record.BackupPath)
		if err != nil {
			return RuntimePatchState{}, fmt.Errorf("read runtime patch backup: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(record.Path), 0o755); err != nil {
			return RuntimePatchState{}, fmt.Errorf("create runtime patch restore dir: %w", err)
		}
		if err := os.WriteFile(record.Path, original, 0o600); err != nil {
			return RuntimePatchState{}, fmt.Errorf("restore runtime patch file: %w", err)
		}
	}
	state.RevertedAt = time.Now().UTC()
	if err := writeJSONFile(statePath, state); err != nil {
		return RuntimePatchState{}, err
	}
	return state, nil
}

func collectProjectFiles(projectRoot string) ([]struct {
	Path     string
	Content  string
	Markdown bool
}, error) {
	files := []struct {
		Path     string
		Content  string
		Markdown bool
	}{}
	err := filepath.WalkDir(projectRoot, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			base := filepath.Base(path)
			if base == ".git" || base == "node_modules" || base == ".venv" || base == "venv" || base == "env" || base == ".env" || base == ".fusion" || base == "knowledgebase" || base == "docs" || base == "vendor" || base == "dist" || base == "build" || base == "__pycache__" || base == "site-packages" {
				return filepath.SkipDir
			}
			if _, err := os.Stat(filepath.Join(path, "pyvenv.cfg")); err == nil {
				return filepath.SkipDir
			}
			return nil
		}
		base := filepath.Base(path)
		ext := strings.ToLower(filepath.Ext(path))
		includeMarkdown := ext == ".md" && (strings.EqualFold(base, "README.md") || strings.HasPrefix(strings.ToUpper(base), "README."))
		if ext != ".py" && ext != ".txt" && ext != ".toml" && ext != ".yaml" && ext != ".yml" && ext != ".json" && !includeMarkdown && base != "requirements.txt" && base != "setup.py" && base != "setup.cfg" {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		files = append(files, struct {
			Path     string
			Content  string
			Markdown bool
		}{
			Path:     path,
			Content:  string(data),
			Markdown: includeMarkdown,
		})
		if len(files) >= 64 {
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk project files: %w", err)
	}
	return files, nil
}

func safeProjectPath(projectRoot, relative string) (string, error) {
	path := strings.TrimSpace(relative)
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(projectRoot, path)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve runtime patch path: %w", err)
	}
	root, err := filepath.Abs(projectRoot)
	if err != nil {
		return "", fmt.Errorf("resolve project root: %w", err)
	}
	if abs != root && !strings.HasPrefix(abs, root+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes project root", relative)
	}
	return abs, nil
}

func contentHash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func writeJSONFile(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create parent dir: %w", err)
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("encode json: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write json: %w", err)
	}
	return nil
}

func readJSONFile(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read json: %w", err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("decode json: %w", err)
	}
	return nil
}
