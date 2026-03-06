package optimize

import (
	"path/filepath"
	"testing"

	"github.com/ModelsLab/fusion/internal/kb"
)

func TestSessionSaveLoadAndStageRecording(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	projectRoot := t.TempDir()

	session := store.NewSession(SessionCreateRequest{
		Name:        "b200-decode",
		ProjectRoot: projectRoot,
		Target:      "lab-b200",
		Runtime:     "vllm",
		Query:       "optimize qwen on b200 for decode",
		Request: Request{
			GPU:       "b200",
			Model:     "qwen2.5-72b",
			Workload:  "decode",
			Operators: []string{"attention", "kv-cache"},
			Precision: "fp8",
		},
		Context: kb.ContextPacket{
			Request: kb.ContextRequest{GPU: "b200", Workload: "decode"},
		},
	})

	if session.WorkspaceRoot != filepath.Join(projectRoot, ".fusion", "optimize", session.ID) {
		t.Fatalf("unexpected workspace root %q", session.WorkspaceRoot)
	}

	candidate := session.UpsertCandidate(Candidate{
		Name:      "triton-attention",
		Backend:   "triton",
		Workspace: filepath.Join(session.WorkspaceRoot, "triton-attention"),
	})
	if err := session.RecordCandidateStage(candidate.ID, "build", "/tmp/build.json", "python build.py", 0, map[string]float64{"compile_status": 1}); err != nil {
		t.Fatalf("RecordCandidateStage() error = %v", err)
	}

	path, err := store.Save(session)
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if path == "" {
		t.Fatal("expected saved session path")
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(loaded.Candidates) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(loaded.Candidates))
	}
	if loaded.Candidates[0].Stages["build"].ArtifactPath != "/tmp/build.json" {
		t.Fatalf("expected build artifact path to round-trip, got %+v", loaded.Candidates[0].Stages["build"])
	}
}

func TestSessionSaveMergesStaleCandidateSnapshots(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	projectRoot := t.TempDir()

	original := store.NewSession(SessionCreateRequest{
		Name:        "merge-candidates",
		ProjectRoot: projectRoot,
		Request: Request{
			GPU:      "b200",
			Workload: "decode",
		},
	})
	if _, err := store.Save(original); err != nil {
		t.Fatalf("Save() initial error = %v", err)
	}

	first, err := store.Load(original.ID)
	if err != nil {
		t.Fatalf("Load() first error = %v", err)
	}
	second, err := store.Load(original.ID)
	if err != nil {
		t.Fatalf("Load() second error = %v", err)
	}

	first.UpsertCandidate(Candidate{
		Name:      "triton-rmsnorm",
		Backend:   "triton",
		Workspace: filepath.Join(original.WorkspaceRoot, "triton-rmsnorm"),
	})
	second.UpsertCandidate(Candidate{
		Name:      "cuda-rmsnorm",
		Backend:   "cuda",
		Workspace: filepath.Join(original.WorkspaceRoot, "cuda-rmsnorm"),
	})

	if _, err := store.Save(first); err != nil {
		t.Fatalf("Save() first snapshot error = %v", err)
	}
	if _, err := store.Save(second); err != nil {
		t.Fatalf("Save() second snapshot error = %v", err)
	}

	loaded, err := store.Load(original.ID)
	if err != nil {
		t.Fatalf("Load() merged error = %v", err)
	}
	if len(loaded.Candidates) != 2 {
		t.Fatalf("expected 2 merged candidates, got %d", len(loaded.Candidates))
	}
	if _, ok := loaded.CandidateByWorkspace("triton", filepath.Join(original.WorkspaceRoot, "triton-rmsnorm")); !ok {
		t.Fatal("expected merged Triton candidate")
	}
	if _, ok := loaded.CandidateByWorkspace("cuda", filepath.Join(original.WorkspaceRoot, "cuda-rmsnorm")); !ok {
		t.Fatal("expected merged CUDA candidate")
	}
}

func TestSessionSaveMergesStaleStageSnapshots(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	projectRoot := t.TempDir()

	session := store.NewSession(SessionCreateRequest{
		Name:        "merge-stages",
		ProjectRoot: projectRoot,
		Request: Request{
			GPU:      "h100",
			Workload: "decode",
		},
	})
	candidate := session.UpsertCandidate(Candidate{
		Name:      "attention",
		Backend:   "triton",
		Workspace: filepath.Join(session.WorkspaceRoot, "triton-attention"),
	})
	if _, err := store.Save(session); err != nil {
		t.Fatalf("Save() initial error = %v", err)
	}

	first, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() first error = %v", err)
	}
	second, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() second error = %v", err)
	}

	if err := first.RecordCandidateStage(candidate.ID, "build", "/tmp/build.json", "python build.py", 0, map[string]float64{"compile_status": 1}); err != nil {
		t.Fatalf("RecordCandidateStage() build error = %v", err)
	}
	if err := second.RecordCandidateStage(candidate.ID, "verify", "/tmp/verify.json", "python verify.py", 0, map[string]float64{"pass": 1}); err != nil {
		t.Fatalf("RecordCandidateStage() verify error = %v", err)
	}

	if _, err := store.Save(first); err != nil {
		t.Fatalf("Save() first stage snapshot error = %v", err)
	}
	if _, err := store.Save(second); err != nil {
		t.Fatalf("Save() second stage snapshot error = %v", err)
	}

	loaded, err := store.Load(session.ID)
	if err != nil {
		t.Fatalf("Load() merged error = %v", err)
	}
	mergedCandidate, ok := loaded.CandidateByID(candidate.ID)
	if !ok {
		t.Fatalf("expected candidate %s", candidate.ID)
	}
	if len(mergedCandidate.Stages) != 2 {
		t.Fatalf("expected 2 merged stages, got %d", len(mergedCandidate.Stages))
	}
	if mergedCandidate.Stages["build"].ArtifactPath != "/tmp/build.json" {
		t.Fatalf("expected merged build stage, got %+v", mergedCandidate.Stages["build"])
	}
	if mergedCandidate.Stages["verify"].ArtifactPath != "/tmp/verify.json" {
		t.Fatalf("expected merged verify stage, got %+v", mergedCandidate.Stages["verify"])
	}
}

func TestSessionPathSanitizesIDs(t *testing.T) {
	store := &SessionStore{root: t.TempDir()}
	path := store.SessionPath("../../etc/passwd")
	if filepath.Dir(path) != store.root {
		t.Fatalf("expected session path to stay inside the store root, got %q", path)
	}
	if filepath.Base(path) == "passwd.json" && filepath.Clean(path) != filepath.Join(store.root, "passwd.json") {
		t.Fatalf("expected sanitized filename, got %q", path)
	}
	if filepath.Base(path) == "../../etc/passwd.json" {
		t.Fatalf("expected sanitized filename, got %q", path)
	}
}
