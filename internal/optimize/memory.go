package optimize

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type SessionMemoryEntry struct {
	Title       string             `json:"title"`
	Category    string             `json:"category,omitempty"`
	Summary     string             `json:"summary"`
	Outcome     string             `json:"outcome,omitempty"`
	CandidateID string             `json:"candidate_id,omitempty"`
	Metrics     map[string]float64 `json:"metrics,omitempty"`
	Lessons     []string           `json:"lessons,omitempty"`
	NextSteps   []string           `json:"next_steps,omitempty"`
	Files       []string           `json:"files,omitempty"`
	RecordedAt  time.Time          `json:"recorded_at"`
}

func SessionMemoryDir(session *Session) string {
	if session == nil {
		return ""
	}
	return filepath.Join(session.WorkspaceRoot, "memory")
}

func SessionMemoryIndexPath(session *Session) string {
	return filepath.Join(SessionMemoryDir(session), "README.md")
}

func SaveSessionMemoryEntry(session *Session, entry SessionMemoryEntry) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	if strings.TrimSpace(entry.Title) == "" {
		return "", fmt.Errorf("memory title is required")
	}
	if entry.RecordedAt.IsZero() {
		entry.RecordedAt = time.Now().UTC()
	}
	dir := SessionMemoryDir(session)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create session memory dir: %w", err)
	}
	filename := entry.RecordedAt.Format("20060102-150405") + "-" + safeSessionPathID(entry.Title) + ".md"
	path := filepath.Join(dir, filename)
	content := renderSessionMemoryEntry(session, entry)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		return "", fmt.Errorf("write session memory entry: %w", err)
	}
	if _, err := RefreshSessionMemoryIndex(session); err != nil {
		return "", err
	}
	return path, nil
}

func RefreshSessionMemoryIndex(session *Session) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	dir := SessionMemoryDir(session)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create session memory dir: %w", err)
	}
	path := SessionMemoryIndexPath(session)
	content, err := renderSessionMemoryIndex(session)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		return "", fmt.Errorf("write session memory index: %w", err)
	}
	return path, nil
}

func renderSessionMemoryEntry(session *Session, entry SessionMemoryEntry) string {
	lines := []string{
		"# " + strings.TrimSpace(entry.Title),
		"",
		fmt.Sprintf("- Session: `%s`", session.ID),
		fmt.Sprintf("- Recorded At: %s", entry.RecordedAt.Format(time.RFC3339)),
	}
	if strings.TrimSpace(entry.Category) != "" {
		lines = append(lines, fmt.Sprintf("- Category: %s", strings.TrimSpace(entry.Category)))
	}
	if strings.TrimSpace(entry.CandidateID) != "" {
		lines = append(lines, fmt.Sprintf("- Candidate: `%s`", strings.TrimSpace(entry.CandidateID)))
	}
	if strings.TrimSpace(entry.Outcome) != "" {
		lines = append(lines, fmt.Sprintf("- Outcome: %s", strings.TrimSpace(entry.Outcome)))
	}
	lines = append(lines, "", "## Summary", "", strings.TrimSpace(entry.Summary))
	if len(entry.Metrics) > 0 {
		lines = append(lines, "", "## Metrics", "")
		keys := make([]string, 0, len(entry.Metrics))
		for key := range entry.Metrics {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			lines = append(lines, fmt.Sprintf("- `%s`: %.6f", key, entry.Metrics[key]))
		}
	}
	if len(entry.Lessons) > 0 {
		lines = append(lines, "", "## Lessons", "")
		for _, lesson := range entry.Lessons {
			lines = append(lines, "- "+strings.TrimSpace(lesson))
		}
	}
	if len(entry.NextSteps) > 0 {
		lines = append(lines, "", "## Next Steps", "")
		for _, step := range entry.NextSteps {
			lines = append(lines, "- "+strings.TrimSpace(step))
		}
	}
	if len(entry.Files) > 0 {
		lines = append(lines, "", "## Files", "")
		for _, file := range entry.Files {
			lines = append(lines, "- `"+strings.TrimSpace(file)+"`")
		}
	}
	lines = append(lines, "")
	return strings.Join(lines, "\n")
}

func renderSessionMemoryIndex(session *Session) (string, error) {
	dirEntries, err := os.ReadDir(SessionMemoryDir(session))
	if err != nil && !os.IsNotExist(err) {
		return "", fmt.Errorf("read session memory dir: %w", err)
	}
	entries := []string{}
	for _, entry := range dirEntries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".md" || entry.Name() == "README.md" {
			continue
		}
		entries = append(entries, entry.Name())
	}
	sort.Sort(sort.Reverse(sort.StringSlice(entries)))

	lines := []string{
		"# Fusion Session Memory",
		"",
		fmt.Sprintf("- Session: `%s`", session.ID),
		fmt.Sprintf("- Name: %s", session.Name),
		fmt.Sprintf("- Project Root: `%s`", session.ProjectRoot),
		fmt.Sprintf("- Workspace Root: `%s`", session.WorkspaceRoot),
		fmt.Sprintf("- Runtime: %s", valueOrFallback(session.Runtime, "unspecified")),
		fmt.Sprintf("- Task: %s", valueOrFallback(session.Request.Task, "unspecified")),
		fmt.Sprintf("- Workload: %s", valueOrFallback(session.Request.Workload, "unspecified")),
		fmt.Sprintf("- Model: %s", valueOrFallback(session.Request.Model, "unspecified")),
		fmt.Sprintf("- GPU: %s", valueOrFallback(session.Request.GPU, "unspecified")),
		fmt.Sprintf("- Current Best Candidate: %s", valueOrFallback(session.CurrentBestID, "unset")),
		"",
		"## Loop Decisions",
		"",
	}
	if len(session.LoopDecisions) == 0 {
		lines = append(lines, "- none recorded")
	} else {
		decisions := append([]LoopDecision{}, session.LoopDecisions...)
		sort.Slice(decisions, func(i, j int) bool {
			return decisions[i].UpdatedAt.After(decisions[j].UpdatedAt)
		})
		for _, decision := range decisions {
			line := fmt.Sprintf("- `%s/%s`: %s", decision.Phase, decision.Family, decision.Status)
			if strings.TrimSpace(decision.CandidateID) != "" {
				line += fmt.Sprintf(" (`%s`)", decision.CandidateID)
			}
			if strings.TrimSpace(decision.Reason) != "" {
				line += " - " + strings.TrimSpace(decision.Reason)
			}
			lines = append(lines, line)
		}
	}

	lines = append(lines, "", "## Candidates", "")
	if len(session.Candidates) == 0 {
		lines = append(lines, "- none registered")
	} else {
		candidates := append([]Candidate{}, session.Candidates...)
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].UpdatedAt.After(candidates[j].UpdatedAt)
		})
		for _, candidate := range candidates {
			line := fmt.Sprintf("- `%s` [%s]", candidate.ID, candidate.Backend)
			if candidate.Winner {
				line += " winner"
			}
			if strings.TrimSpace(candidate.Status) != "" {
				line += " status=" + candidate.Status
			}
			if candidate.Score != 0 {
				line += fmt.Sprintf(" score=%.4f", candidate.Score)
			}
			lines = append(lines, line)
		}
	}

	lines = append(lines, "", "## Memory Entries", "")
	if len(entries) == 0 {
		lines = append(lines, "- none recorded")
	} else {
		for _, entry := range entries {
			lines = append(lines, fmt.Sprintf("- [%s](./%s)", strings.TrimSuffix(entry, ".md"), entry))
		}
	}
	lines = append(lines, "")
	return strings.Join(lines, "\n"), nil
}
