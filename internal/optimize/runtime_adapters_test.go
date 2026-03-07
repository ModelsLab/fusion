package optimize

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectRuntimeAdapters(t *testing.T) {
	root := t.TempDir()
	source := `from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("foo")`
	if err := os.WriteFile(filepath.Join(root, "app.py"), []byte(source), 0o600); err != nil {
		t.Fatalf("write app.py: %v", err)
	}
	detections, err := DetectRuntimeAdapters(root)
	if err != nil {
		t.Fatalf("DetectRuntimeAdapters() error = %v", err)
	}
	found := false
	for _, detection := range detections {
		if detection.Adapter == "diffusers" && detection.Matched {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected diffusers adapter to be detected, got %+v", detections)
	}
}

func TestDetectRuntimeAdaptersIgnoresDocsOnlyMentions(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "knowledgebase", "documents"), 0o755); err != nil {
		t.Fatalf("create knowledgebase docs dir: %v", err)
	}
	doc := `# Notes

This mentions diffusers and transformers, but it is not project code.`
	if err := os.WriteFile(filepath.Join(root, "knowledgebase", "documents", "notes.md"), []byte(doc), 0o600); err != nil {
		t.Fatalf("write docs note: %v", err)
	}
	detections, err := DetectRuntimeAdapters(root)
	if err != nil {
		t.Fatalf("DetectRuntimeAdapters() error = %v", err)
	}
	for _, detection := range detections {
		if detection.Adapter == "diffusers" && detection.Matched {
			t.Fatalf("expected docs-only diffusers mention to be ignored, got %+v", detections)
		}
		if detection.Adapter == "transformers" && detection.Matched {
			t.Fatalf("expected docs-only transformers mention to be ignored, got %+v", detections)
		}
	}
}

func TestDetectRuntimeAdaptersIgnoresVirtualenvDirectories(t *testing.T) {
	root := t.TempDir()
	venv := filepath.Join(root, "demoenv")
	if err := os.MkdirAll(filepath.Join(venv, "lib"), 0o755); err != nil {
		t.Fatalf("create venv dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(venv, "pyvenv.cfg"), []byte("home = /usr/bin\n"), 0o600); err != nil {
		t.Fatalf("write pyvenv.cfg: %v", err)
	}
	source := `from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("foo")`
	if err := os.WriteFile(filepath.Join(venv, "site.py"), []byte(source), 0o600); err != nil {
		t.Fatalf("write site.py: %v", err)
	}
	detections, err := DetectRuntimeAdapters(root)
	if err != nil {
		t.Fatalf("DetectRuntimeAdapters() error = %v", err)
	}
	for _, detection := range detections {
		if detection.Adapter == "diffusers" && detection.Matched {
			t.Fatalf("expected virtualenv contents to be ignored, got %+v", detections)
		}
	}
}

func TestApplyAndRevertRuntimePatch(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "pipeline.py")
	if err := os.WriteFile(target, []byte("print('before')\n"), 0o600); err != nil {
		t.Fatalf("write target: %v", err)
	}
	state, statePath, err := ApplyRuntimePatch(RuntimePatchPlan{
		Version:     1,
		Adapter:     "generic-python",
		ProjectRoot: root,
		Operations: []RuntimePatchOperation{
			{Path: "pipeline.py", Content: "print('after')\n"},
		},
	})
	if err != nil {
		t.Fatalf("ApplyRuntimePatch() error = %v", err)
	}
	if len(state.Records) != 1 {
		t.Fatalf("expected one runtime patch record, got %+v", state)
	}
	updated, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read updated target: %v", err)
	}
	if string(updated) != "print('after')\n" {
		t.Fatalf("expected updated file contents, got %q", string(updated))
	}
	if _, err := RevertRuntimePatch(statePath); err != nil {
		t.Fatalf("RevertRuntimePatch() error = %v", err)
	}
	reverted, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read reverted target: %v", err)
	}
	if string(reverted) != "print('before')\n" {
		t.Fatalf("expected reverted file contents, got %q", string(reverted))
	}
}
