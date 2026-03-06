package optimize

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/kb"
)

const sessionVersion = 1

type Session struct {
	Version        int              `json:"version"`
	ID             string           `json:"id"`
	Name           string           `json:"name"`
	ProjectRoot    string           `json:"project_root"`
	WorkspaceRoot  string           `json:"workspace_root"`
	Target         string           `json:"target,omitempty"`
	Runtime        string           `json:"runtime,omitempty"`
	Query          string           `json:"query,omitempty"`
	Status         string           `json:"status,omitempty"`
	AgentSessionID string           `json:"agent_session_id,omitempty"`
	LastResult     string           `json:"last_result,omitempty"`
	CurrentBestID  string           `json:"current_best_id,omitempty"`
	LoopDecisions  []LoopDecision   `json:"loop_decisions,omitempty"`
	InnerLoop      InnerLoopState   `json:"inner_loop,omitempty"`
	Request        Request          `json:"request"`
	Context        kb.ContextPacket `json:"context"`
	Candidates     []Candidate      `json:"candidates,omitempty"`
	Notes          []string         `json:"notes,omitempty"`
	CreatedAt      time.Time        `json:"created_at"`
	UpdatedAt      time.Time        `json:"updated_at"`
}

type Candidate struct {
	ID          string                    `json:"id"`
	Name        string                    `json:"name"`
	Backend     string                    `json:"backend"`
	Template    string                    `json:"template,omitempty"`
	Operation   string                    `json:"operation,omitempty"`
	GPUArch     string                    `json:"gpu_arch,omitempty"`
	Workspace   string                    `json:"workspace"`
	ParentID    string                    `json:"parent_id,omitempty"`
	Status      string                    `json:"status,omitempty"`
	SearchMode  string                    `json:"search_mode,omitempty"`
	SearchLane  string                    `json:"search_lane,omitempty"`
	Round       int                       `json:"round,omitempty"`
	Hypothesis  string                    `json:"hypothesis,omitempty"`
	Score       float64                   `json:"score,omitempty"`
	Winner      bool                      `json:"winner,omitempty"`
	RejectReason string                   `json:"reject_reason,omitempty"`
	CreatedAt   time.Time                 `json:"created_at"`
	UpdatedAt   time.Time                 `json:"updated_at"`
	Stages      map[string]CandidateStage `json:"stages,omitempty"`
	Description string                    `json:"description,omitempty"`
}

type CandidateStage struct {
	ArtifactPath string             `json:"artifact_path,omitempty"`
	Command      string             `json:"command,omitempty"`
	ExitCode     int                `json:"exit_code"`
	Metrics      map[string]float64 `json:"metrics,omitempty"`
	UpdatedAt    time.Time          `json:"updated_at"`
}

type LoopDecision struct {
	Phase      string    `json:"phase"`
	Family     string    `json:"family"`
	Status     string    `json:"status"`
	CandidateID string   `json:"candidate_id,omitempty"`
	Reason     string    `json:"reason,omitempty"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type InnerLoopState struct {
	Status        string    `json:"status,omitempty"`
	SearchMode    string    `json:"search_mode,omitempty"`
	BeamWidth     int       `json:"beam_width,omitempty"`
	CurrentRound  int       `json:"current_round,omitempty"`
	BestCandidateID string  `json:"best_candidate_id,omitempty"`
	StartedAt     time.Time `json:"started_at,omitempty"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
}

type SessionSummary struct {
	ID             string    `json:"id"`
	Name           string    `json:"name"`
	ProjectRoot    string    `json:"project_root"`
	WorkspaceRoot  string    `json:"workspace_root"`
	GPU            string    `json:"gpu,omitempty"`
	Workload       string    `json:"workload,omitempty"`
	Runtime        string    `json:"runtime,omitempty"`
	Status         string    `json:"status,omitempty"`
	CandidateCount int       `json:"candidate_count"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type SessionCreateRequest struct {
	Name          string
	ProjectRoot   string
	WorkspaceRoot string
	Target        string
	Runtime       string
	Query         string
	Request       Request
	Context       kb.ContextPacket
	Notes         []string
}

type SessionStore struct {
	root string
}

const (
	sessionLockWait   = 10 * time.Second
	sessionLockPoll   = 50 * time.Millisecond
	sessionLockMaxAge = 5 * time.Minute
)

func NewSessionStore() (*SessionStore, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("resolve user config dir: %w", err)
	}

	return &SessionStore{
		root: filepath.Join(configDir, "fusion", "optimize", "sessions"),
	}, nil
}

func (s *SessionStore) SessionPath(id string) string {
	return filepath.Join(s.root, safeSessionPathID(id)+".json")
}

func (s *SessionStore) NewSession(req SessionCreateRequest) *Session {
	projectRoot := strings.TrimSpace(req.ProjectRoot)
	if projectRoot == "" {
		projectRoot = "."
	}
	now := time.Now().UTC()
	id := newSessionID(req.Name, projectRoot)
	workspaceRoot := strings.TrimSpace(req.WorkspaceRoot)
	if workspaceRoot == "" {
		workspaceRoot = DefaultWorkspaceRoot(projectRoot, id)
	}

	name := strings.TrimSpace(req.Name)
	if name == "" {
		name = filepath.Base(projectRoot)
		if name == "." || name == string(filepath.Separator) || name == "" {
			name = "optimization-session"
		}
	}

	return &Session{
		Version:       sessionVersion,
		ID:            id,
		Name:          name,
		ProjectRoot:   projectRoot,
		WorkspaceRoot: workspaceRoot,
		Target:        strings.TrimSpace(req.Target),
		Runtime:       strings.TrimSpace(req.Runtime),
		Query:         strings.TrimSpace(req.Query),
		Request:       req.Request,
		Context:       req.Context,
		Notes:         append([]string{}, req.Notes...),
		CreatedAt:     now,
		UpdatedAt:     now,
	}
}

func (s *SessionStore) Save(session *Session) (string, error) {
	if session == nil {
		return "", fmt.Errorf("session is required")
	}
	input := session
	session.ID = safeSessionPathID(session.ID)
	if session.Version == 0 {
		session.Version = sessionVersion
	}
	session.UpdatedAt = time.Now().UTC()

	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return "", fmt.Errorf("create optimization sessions dir: %w", err)
	}
	lockPath := s.SessionPath(session.ID) + ".lock"
	unlock, err := acquireSessionLock(lockPath)
	if err != nil {
		return "", err
	}
	defer unlock()

	path := s.SessionPath(session.ID)
	existing, err := s.loadPath(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	if existing != nil {
		session = mergeSessions(existing, session)
	}
	if err := os.MkdirAll(session.WorkspaceRoot, 0o755); err != nil {
		return "", fmt.Errorf("create session workspace root: %w", err)
	}

	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode optimization session: %w", err)
	}
	data = append(data, '\n')

	if err := os.WriteFile(path, data, 0o600); err != nil {
		return "", fmt.Errorf("write optimization session: %w", err)
	}
	*input = *session
	return path, nil
}

func (s *SessionStore) Load(id string) (*Session, error) {
	return s.loadPath(s.SessionPath(id))
}

func (s *SessionStore) List() ([]SessionSummary, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read optimization sessions dir: %w", err)
	}

	summaries := []SessionSummary{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.root, entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("read optimization session summary: %w", err)
		}
		var session Session
		if err := json.Unmarshal(data, &session); err != nil {
			return nil, fmt.Errorf("decode optimization session summary: %w", err)
		}
		summaries = append(summaries, SessionSummary{
			ID:             session.ID,
			Name:           session.Name,
			ProjectRoot:    session.ProjectRoot,
			WorkspaceRoot:  session.WorkspaceRoot,
			GPU:            session.Request.GPU,
			Workload:       session.Request.Workload,
			Runtime:        session.Runtime,
			Status:         session.Status,
			CandidateCount: len(session.Candidates),
			UpdatedAt:      session.UpdatedAt,
		})
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].UpdatedAt.After(summaries[j].UpdatedAt)
	})
	return summaries, nil
}

func DefaultWorkspaceRoot(projectRoot, id string) string {
	return filepath.Join(projectRoot, ".fusion", "optimize", id)
}

func (s *Session) UpsertCandidate(candidate Candidate) Candidate {
	now := time.Now().UTC()
	if candidate.ID == "" {
		candidate.ID = newCandidateID(candidate.Backend, candidate.Name, candidate.Workspace)
	}
	if candidate.Name == "" {
		candidate.Name = filepath.Base(candidate.Workspace)
	}
	if candidate.CreatedAt.IsZero() {
		candidate.CreatedAt = now
	}
	candidate.UpdatedAt = now
	if candidate.Stages == nil {
		candidate.Stages = map[string]CandidateStage{}
	}

	for i := range s.Candidates {
		if s.Candidates[i].ID == candidate.ID || (s.Candidates[i].Backend == candidate.Backend && s.Candidates[i].Workspace == candidate.Workspace) {
			candidate.CreatedAt = s.Candidates[i].CreatedAt
			if len(candidate.Stages) == 0 {
				candidate.Stages = s.Candidates[i].Stages
			}
			if strings.TrimSpace(candidate.ParentID) == "" {
				candidate.ParentID = s.Candidates[i].ParentID
			}
			if strings.TrimSpace(candidate.Status) == "" {
				candidate.Status = s.Candidates[i].Status
			}
			if strings.TrimSpace(candidate.SearchMode) == "" {
				candidate.SearchMode = s.Candidates[i].SearchMode
			}
			if strings.TrimSpace(candidate.SearchLane) == "" {
				candidate.SearchLane = s.Candidates[i].SearchLane
			}
			if candidate.Round == 0 {
				candidate.Round = s.Candidates[i].Round
			}
			if strings.TrimSpace(candidate.Hypothesis) == "" {
				candidate.Hypothesis = s.Candidates[i].Hypothesis
			}
			if candidate.Score == 0 {
				candidate.Score = s.Candidates[i].Score
			}
			if !candidate.Winner {
				candidate.Winner = s.Candidates[i].Winner
			}
			if strings.TrimSpace(candidate.RejectReason) == "" {
				candidate.RejectReason = s.Candidates[i].RejectReason
			}
			s.Candidates[i] = candidate
			s.UpdatedAt = now
			return candidate
		}
	}

	s.Candidates = append(s.Candidates, candidate)
	s.UpdatedAt = now
	return candidate
}

func (s *Session) RecordCandidateStage(candidateID, stage, artifactPath, command string, exitCode int, metrics map[string]float64) error {
	stage = strings.TrimSpace(strings.ToLower(stage))
	if stage == "" {
		return fmt.Errorf("stage is required")
	}
	now := time.Now().UTC()
	for i := range s.Candidates {
		if s.Candidates[i].ID != candidateID {
			continue
		}
		if s.Candidates[i].Stages == nil {
			s.Candidates[i].Stages = map[string]CandidateStage{}
		}
		s.Candidates[i].Stages[stage] = CandidateStage{
			ArtifactPath: strings.TrimSpace(artifactPath),
			Command:      strings.TrimSpace(command),
			ExitCode:     exitCode,
			Metrics:      cloneMetrics(metrics),
			UpdatedAt:    now,
		}
		s.Candidates[i].UpdatedAt = now
		s.UpdatedAt = now
		return nil
	}
	return fmt.Errorf("candidate %q not found in session %s", candidateID, s.ID)
}

func (s *Session) RecordLoopDecision(phase, family, status, candidateID, reason string) {
	now := time.Now().UTC()
	decision := LoopDecision{
		Phase:       canonicalLoopValue(phase),
		Family:      canonicalLoopValue(family),
		Status:      canonicalLoopValue(status),
		CandidateID: strings.TrimSpace(candidateID),
		Reason:      strings.TrimSpace(reason),
		UpdatedAt:   now,
	}
	for i := range s.LoopDecisions {
		if s.LoopDecisions[i].Phase == decision.Phase && s.LoopDecisions[i].Family == decision.Family {
			s.LoopDecisions[i] = decision
			s.UpdatedAt = now
			return
		}
	}
	s.LoopDecisions = append(s.LoopDecisions, decision)
	s.UpdatedAt = now
}

func (s *Session) LoopDecision(phase, family string) (LoopDecision, bool) {
	phase = canonicalLoopValue(phase)
	family = canonicalLoopValue(family)
	for _, decision := range s.LoopDecisions {
		if decision.Phase == phase && decision.Family == family {
			return decision, true
		}
	}
	return LoopDecision{}, false
}

func (s *Session) SetCurrentBestCandidate(candidateID string) {
	candidateID = strings.TrimSpace(candidateID)
	s.CurrentBestID = candidateID
	now := time.Now().UTC()
	for i := range s.Candidates {
		s.Candidates[i].Winner = s.Candidates[i].ID == candidateID && candidateID != ""
		if s.Candidates[i].Winner {
			s.Candidates[i].Status = "winner"
			s.Candidates[i].UpdatedAt = now
		}
	}
	s.UpdatedAt = now
}

func (s *Session) CandidateByID(id string) (*Candidate, bool) {
	target := strings.TrimSpace(id)
	for i := range s.Candidates {
		if s.Candidates[i].ID == target {
			return &s.Candidates[i], true
		}
	}
	return nil, false
}

func (s *Session) CandidateByWorkspace(backend, workspace string) (*Candidate, bool) {
	backend = strings.TrimSpace(strings.ToLower(backend))
	workspace = filepath.Clean(strings.TrimSpace(workspace))
	for i := range s.Candidates {
		if strings.TrimSpace(strings.ToLower(s.Candidates[i].Backend)) == backend && filepath.Clean(s.Candidates[i].Workspace) == workspace {
			return &s.Candidates[i], true
		}
	}
	return nil, false
}

func newSessionID(name, projectRoot string) string {
	base := strings.TrimSpace(name)
	if base == "" {
		base = filepath.Base(projectRoot)
	}
	base = sanitizeID(base)
	if base == "" {
		base = "optimize"
	}
	return time.Now().UTC().Format("20060102-150405") + "-" + base
}

func newCandidateID(backend, name, workspace string) string {
	base := strings.TrimSpace(name)
	if base == "" {
		base = filepath.Base(strings.TrimSpace(workspace))
	}
	base = sanitizeID(strings.TrimSpace(backend) + "-" + base)
	if base == "" {
		base = "candidate"
	}
	return base
}

func safeSessionPathID(value string) string {
	value = sanitizeID(value)
	if value == "" {
		return "session"
	}
	return value
}

func sanitizeID(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	value = strings.NewReplacer(" ", "-", "/", "-", "_", "-", ".", "-").Replace(value)
	value = strings.Trim(value, "-")
	return value
}

func cloneMetrics(metrics map[string]float64) map[string]float64 {
	if len(metrics) == 0 {
		return nil
	}
	out := make(map[string]float64, len(metrics))
	for key, value := range metrics {
		out[key] = value
	}
	return out
}

func (s *SessionStore) loadPath(path string) (*Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read optimization session: %w", err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("decode optimization session: %w", err)
	}
	return &session, nil
}

func acquireSessionLock(path string) (func(), error) {
	deadline := time.Now().Add(sessionLockWait)
	for {
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = file.WriteString(strconv.FormatInt(time.Now().UTC().Unix(), 10))
			_ = file.Close()
			return func() {
				_ = os.Remove(path)
			}, nil
		}
		if !os.IsExist(err) {
			return nil, fmt.Errorf("create session lock: %w", err)
		}
		if stale, staleErr := staleSessionLock(path); staleErr == nil && stale {
			_ = os.Remove(path)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for session lock %s", path)
		}
		time.Sleep(sessionLockPoll)
	}
}

func staleSessionLock(path string) (bool, error) {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return time.Since(info.ModTime()) > sessionLockMaxAge, nil
}

func mergeSessions(existing, incoming *Session) *Session {
	if existing == nil {
		return cloneSession(incoming)
	}
	if incoming == nil {
		return cloneSession(existing)
	}

	merged := cloneSession(incoming)
	merged.Version = nonZeroInt(merged.Version, existing.Version)
	merged.ID = valueOrFallback(merged.ID, existing.ID)
	merged.Name = valueOrFallback(merged.Name, existing.Name)
	merged.ProjectRoot = valueOrFallback(merged.ProjectRoot, existing.ProjectRoot)
	merged.WorkspaceRoot = valueOrFallback(merged.WorkspaceRoot, existing.WorkspaceRoot)
	merged.Target = valueOrFallback(merged.Target, existing.Target)
	merged.Runtime = valueOrFallback(merged.Runtime, existing.Runtime)
	merged.Query = valueOrFallback(merged.Query, existing.Query)
	merged.Status = valueOrFallback(merged.Status, existing.Status)
	merged.AgentSessionID = valueOrFallback(merged.AgentSessionID, existing.AgentSessionID)
	merged.LastResult = valueOrFallback(merged.LastResult, existing.LastResult)
	merged.CurrentBestID = valueOrFallback(merged.CurrentBestID, existing.CurrentBestID)
	if requestEmpty(merged.Request) {
		merged.Request = cloneRequest(existing.Request)
	}
	if contextPacketEmpty(merged.Context) {
		merged.Context = cloneContext(existing.Context)
	}
	merged.Notes = mergeStrings(existing.Notes, merged.Notes)
	merged.LoopDecisions = mergeLoopDecisions(existing.LoopDecisions, merged.LoopDecisions)
	merged.InnerLoop = mergeInnerLoop(existing.InnerLoop, merged.InnerLoop)
	merged.Candidates = mergeCandidates(existing.Candidates, merged.Candidates)
	merged.CreatedAt = chooseCreatedAt(existing.CreatedAt, merged.CreatedAt)
	merged.UpdatedAt = laterTime(existing.UpdatedAt, merged.UpdatedAt)
	return merged
}

func mergeCandidates(existing, incoming []Candidate) []Candidate {
	if len(existing) == 0 {
		return cloneCandidates(incoming)
	}
	if len(incoming) == 0 {
		return cloneCandidates(existing)
	}

	merged := cloneCandidates(existing)
	for _, candidate := range incoming {
		index := findCandidateIndex(merged, candidate)
		if index >= 0 {
			merged[index] = mergeCandidate(merged[index], candidate)
			continue
		}
		merged = append(merged, cloneCandidate(candidate))
	}
	return merged
}

func mergeCandidate(existing, incoming Candidate) Candidate {
	merged := cloneCandidate(incoming)
	merged.ID = valueOrFallback(merged.ID, existing.ID)
	merged.Name = valueOrFallback(merged.Name, existing.Name)
	merged.Backend = valueOrFallback(merged.Backend, existing.Backend)
	merged.Template = valueOrFallback(merged.Template, existing.Template)
	merged.Operation = valueOrFallback(merged.Operation, existing.Operation)
	merged.GPUArch = valueOrFallback(merged.GPUArch, existing.GPUArch)
	merged.Workspace = valueOrFallback(merged.Workspace, existing.Workspace)
	merged.ParentID = valueOrFallback(merged.ParentID, existing.ParentID)
	merged.Status = valueOrFallback(merged.Status, existing.Status)
	merged.SearchMode = valueOrFallback(merged.SearchMode, existing.SearchMode)
	merged.SearchLane = valueOrFallback(merged.SearchLane, existing.SearchLane)
	if merged.Round == 0 {
		merged.Round = existing.Round
	}
	merged.Hypothesis = valueOrFallback(merged.Hypothesis, existing.Hypothesis)
	if merged.Score == 0 {
		merged.Score = existing.Score
	}
	if !merged.Winner {
		merged.Winner = existing.Winner
	}
	merged.RejectReason = valueOrFallback(merged.RejectReason, existing.RejectReason)
	merged.Description = valueOrFallback(merged.Description, existing.Description)
	merged.CreatedAt = chooseCreatedAt(existing.CreatedAt, merged.CreatedAt)
	merged.UpdatedAt = laterTime(existing.UpdatedAt, merged.UpdatedAt)
	merged.Stages = mergeStages(existing.Stages, merged.Stages)
	return merged
}

func mergeStages(existing, incoming map[string]CandidateStage) map[string]CandidateStage {
	if len(existing) == 0 {
		return cloneStages(incoming)
	}
	if len(incoming) == 0 {
		return cloneStages(existing)
	}

	out := cloneStages(existing)
	for name, stage := range incoming {
		current, ok := out[name]
		if !ok || stage.UpdatedAt.After(current.UpdatedAt) || current.UpdatedAt.IsZero() {
			out[name] = cloneStage(stage)
		}
	}
	return out
}

func findCandidateIndex(candidates []Candidate, target Candidate) int {
	targetID := strings.TrimSpace(target.ID)
	targetBackend := strings.TrimSpace(strings.ToLower(target.Backend))
	targetWorkspace := filepath.Clean(strings.TrimSpace(target.Workspace))
	for i := range candidates {
		if targetID != "" && candidates[i].ID == targetID {
			return i
		}
		if targetBackend != "" && targetWorkspace != "." &&
			strings.TrimSpace(strings.ToLower(candidates[i].Backend)) == targetBackend &&
			filepath.Clean(strings.TrimSpace(candidates[i].Workspace)) == targetWorkspace {
			return i
		}
	}
	return -1
}

func cloneSession(session *Session) *Session {
	if session == nil {
		return nil
	}
	cloned := *session
	cloned.Request = cloneRequest(session.Request)
	cloned.Context = cloneContext(session.Context)
	cloned.Candidates = cloneCandidates(session.Candidates)
	cloned.Notes = append([]string{}, session.Notes...)
	cloned.LoopDecisions = cloneLoopDecisions(session.LoopDecisions)
	cloned.InnerLoop = session.InnerLoop
	return &cloned
}

func cloneRequest(request Request) Request {
	request.Operators = append([]string{}, request.Operators...)
	request.Goals = append([]string{}, request.Goals...)
	return request
}

func cloneContext(packet kb.ContextPacket) kb.ContextPacket {
	packet.Request.Operators = append([]string{}, packet.Request.Operators...)
	packet.Request.Goals = append([]string{}, packet.Request.Goals...)
	if packet.GPU != nil {
		gpu := *packet.GPU
		gpu.Aliases = append([]string{}, packet.GPU.Aliases...)
		gpu.PreferredPrecisions = append([]string{}, packet.GPU.PreferredPrecisions...)
		gpu.ExperimentalPrecisons = append([]string{}, packet.GPU.ExperimentalPrecisons...)
		gpu.Strengths = append([]string{}, packet.GPU.Strengths...)
		gpu.Constraints = append([]string{}, packet.GPU.Constraints...)
		gpu.SourceIDs = append([]string{}, packet.GPU.SourceIDs...)
		packet.GPU = &gpu
	}
	packet.Strategies = cloneStrategyMatches(packet.Strategies)
	packet.Skills = cloneSkillMatches(packet.Skills)
	packet.Examples = cloneExampleMatches(packet.Examples)
	packet.Documents = cloneDocumentMatches(packet.Documents)
	packet.Sources = cloneSources(packet.Sources)
	packet.Notes = append([]string{}, packet.Notes...)
	return packet
}

func cloneStrategyMatches(matches []kb.StrategyMatch) []kb.StrategyMatch {
	if len(matches) == 0 {
		return nil
	}
	out := make([]kb.StrategyMatch, 0, len(matches))
	for _, match := range matches {
		cloned := match
		cloned.Reasons = append([]string{}, match.Reasons...)
		cloned.Strategy.Workloads = append([]string{}, match.Strategy.Workloads...)
		cloned.Strategy.Operators = append([]string{}, match.Strategy.Operators...)
		cloned.Strategy.GPUFamilies = append([]string{}, match.Strategy.GPUFamilies...)
		cloned.Strategy.GPUIDs = append([]string{}, match.Strategy.GPUIDs...)
		cloned.Strategy.Precision = append([]string{}, match.Strategy.Precision...)
		cloned.Strategy.Bottlenecks = append([]string{}, match.Strategy.Bottlenecks...)
		cloned.Strategy.Goals = append([]string{}, match.Strategy.Goals...)
		cloned.Strategy.Preconditions = append([]string{}, match.Strategy.Preconditions...)
		cloned.Strategy.Actions = append([]string{}, match.Strategy.Actions...)
		cloned.Strategy.Metrics = append([]string{}, match.Strategy.Metrics...)
		cloned.Strategy.Tradeoffs = append([]string{}, match.Strategy.Tradeoffs...)
		cloned.Strategy.SourceIDs = append([]string{}, match.Strategy.SourceIDs...)
		cloned.Sources = cloneSources(match.Sources)
		out = append(out, cloned)
	}
	return out
}

func cloneSkillMatches(matches []kb.SkillMatch) []kb.SkillMatch {
	if len(matches) == 0 {
		return nil
	}
	out := make([]kb.SkillMatch, 0, len(matches))
	for _, match := range matches {
		cloned := match
		cloned.Reasons = append([]string{}, match.Reasons...)
		cloned.Skill.Triggers.GPUFamilies = append([]string{}, match.Skill.Triggers.GPUFamilies...)
		cloned.Skill.Triggers.GPUIDs = append([]string{}, match.Skill.Triggers.GPUIDs...)
		cloned.Skill.Triggers.Workloads = append([]string{}, match.Skill.Triggers.Workloads...)
		cloned.Skill.Triggers.Operators = append([]string{}, match.Skill.Triggers.Operators...)
		cloned.Skill.Triggers.Precision = append([]string{}, match.Skill.Triggers.Precision...)
		cloned.Skill.Triggers.Bottlenecks = append([]string{}, match.Skill.Triggers.Bottlenecks...)
		cloned.Skill.Triggers.Runtimes = append([]string{}, match.Skill.Triggers.Runtimes...)
		cloned.Skill.Triggers.Goals = append([]string{}, match.Skill.Triggers.Goals...)
		cloned.Skill.PreferredBackends = append([]string{}, match.Skill.PreferredBackends...)
		cloned.Skill.RequiredTools = append([]string{}, match.Skill.RequiredTools...)
		cloned.Skill.Steps = append([]string{}, match.Skill.Steps...)
		cloned.Skill.Verification = append([]string{}, match.Skill.Verification...)
		cloned.Skill.BenchmarkRubric = append([]string{}, match.Skill.BenchmarkRubric...)
		cloned.Skill.FailureRecovery = append([]string{}, match.Skill.FailureRecovery...)
		cloned.Skill.ArtifactsToSave = append([]string{}, match.Skill.ArtifactsToSave...)
		cloned.Skill.RuntimeAdapters = append([]string{}, match.Skill.RuntimeAdapters...)
		cloned.Skill.ReferenceSourceIDs = append([]string{}, match.Skill.ReferenceSourceIDs...)
		cloned.Sources = cloneSources(match.Sources)
		out = append(out, cloned)
	}
	return out
}

func cloneExampleMatches(matches []kb.ExampleMatch) []kb.ExampleMatch {
	if len(matches) == 0 {
		return nil
	}
	out := make([]kb.ExampleMatch, 0, len(matches))
	for _, match := range matches {
		cloned := match
		cloned.Reasons = append([]string{}, match.Reasons...)
		cloned.Example.GPUFamilies = append([]string{}, match.Example.GPUFamilies...)
		cloned.Example.GPUIDs = append([]string{}, match.Example.GPUIDs...)
		cloned.Example.Workloads = append([]string{}, match.Example.Workloads...)
		cloned.Example.Operators = append([]string{}, match.Example.Operators...)
		cloned.Example.Precision = append([]string{}, match.Example.Precision...)
		cloned.Example.Runtimes = append([]string{}, match.Example.Runtimes...)
		cloned.Example.UseCases = append([]string{}, match.Example.UseCases...)
		cloned.Example.Notes = append([]string{}, match.Example.Notes...)
		cloned.Example.ReferencePaths = append([]string{}, match.Example.ReferencePaths...)
		cloned.Example.SourceIDs = append([]string{}, match.Example.SourceIDs...)
		cloned.Sources = cloneSources(match.Sources)
		out = append(out, cloned)
	}
	return out
}

func cloneDocumentMatches(matches []kb.DocumentMatch) []kb.DocumentMatch {
	if len(matches) == 0 {
		return nil
	}
	out := make([]kb.DocumentMatch, 0, len(matches))
	for _, match := range matches {
		cloned := match
		cloned.Reasons = append([]string{}, match.Reasons...)
		cloned.Document.Tags = append([]string{}, match.Document.Tags...)
		cloned.Document.GPUFamilies = append([]string{}, match.Document.GPUFamilies...)
		cloned.Document.GPUIDs = append([]string{}, match.Document.GPUIDs...)
		cloned.Document.Workloads = append([]string{}, match.Document.Workloads...)
		cloned.Document.Operators = append([]string{}, match.Document.Operators...)
		cloned.Document.Precision = append([]string{}, match.Document.Precision...)
		cloned.Document.Runtimes = append([]string{}, match.Document.Runtimes...)
		cloned.Document.Backends = append([]string{}, match.Document.Backends...)
		cloned.Document.SourceIDs = append([]string{}, match.Document.SourceIDs...)
		cloned.Sources = cloneSources(match.Sources)
		out = append(out, cloned)
	}
	return out
}

func cloneSources(sources []kb.Source) []kb.Source {
	if len(sources) == 0 {
		return nil
	}
	out := make([]kb.Source, 0, len(sources))
	for _, source := range sources {
		cloned := source
		cloned.Tags = append([]string{}, source.Tags...)
		out = append(out, cloned)
	}
	return out
}

func cloneCandidates(candidates []Candidate) []Candidate {
	if len(candidates) == 0 {
		return nil
	}
	out := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, cloneCandidate(candidate))
	}
	return out
}

func cloneCandidate(candidate Candidate) Candidate {
	candidate.Stages = cloneStages(candidate.Stages)
	return candidate
}

func cloneLoopDecisions(decisions []LoopDecision) []LoopDecision {
	if len(decisions) == 0 {
		return nil
	}
	out := make([]LoopDecision, 0, len(decisions))
	out = append(out, decisions...)
	return out
}

func cloneStages(stages map[string]CandidateStage) map[string]CandidateStage {
	if len(stages) == 0 {
		return nil
	}
	out := make(map[string]CandidateStage, len(stages))
	for name, stage := range stages {
		out[name] = cloneStage(stage)
	}
	return out
}

func cloneStage(stage CandidateStage) CandidateStage {
	stage.Metrics = cloneMetrics(stage.Metrics)
	return stage
}

func requestEmpty(request Request) bool {
	return strings.TrimSpace(request.GPU) == "" &&
		strings.TrimSpace(request.Model) == "" &&
		strings.TrimSpace(request.Workload) == "" &&
		len(request.Operators) == 0 &&
		strings.TrimSpace(request.Precision) == "" &&
		strings.TrimSpace(request.Bottleneck) == "" &&
		len(request.Goals) == 0 &&
		request.BatchSize == 0 &&
		request.ContextLength == 0 &&
		!request.IncludeExperimental
}

func contextPacketEmpty(packet kb.ContextPacket) bool {
	return packet.GPU == nil &&
		len(packet.Strategies) == 0 &&
		len(packet.Skills) == 0 &&
		len(packet.Examples) == 0 &&
		len(packet.Documents) == 0 &&
		len(packet.Sources) == 0 &&
		len(packet.Notes) == 0 &&
		strings.TrimSpace(packet.Request.Query) == "" &&
		strings.TrimSpace(packet.Request.GPU) == "" &&
		strings.TrimSpace(packet.Request.Model) == "" &&
		strings.TrimSpace(packet.Request.Workload) == ""
}

func mergeStrings(existing, incoming []string) []string {
	if len(existing) == 0 {
		return append([]string{}, incoming...)
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(existing)+len(incoming))
	for _, value := range existing {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	for _, value := range incoming {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	return out
}

func chooseCreatedAt(existing, incoming time.Time) time.Time {
	if existing.IsZero() {
		return incoming
	}
	if incoming.IsZero() || existing.Before(incoming) {
		return existing
	}
	return incoming
}

func laterTime(existing, incoming time.Time) time.Time {
	if existing.After(incoming) {
		return existing
	}
	return incoming
}

func valueOrFallback(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value != "" {
		return value
	}
	return strings.TrimSpace(fallback)
}

func nonZeroInt(value, fallback int) int {
	if value != 0 {
		return value
	}
	return fallback
}

func canonicalLoopValue(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	value = strings.NewReplacer(" ", "-", "_", "-").Replace(value)
	return strings.Trim(value, "-")
}

func mergeLoopDecisions(existing, incoming []LoopDecision) []LoopDecision {
	if len(existing) == 0 {
		return cloneLoopDecisions(incoming)
	}
	if len(incoming) == 0 {
		return cloneLoopDecisions(existing)
	}
	out := cloneLoopDecisions(existing)
	for _, decision := range incoming {
		replaced := false
		for i := range out {
			if out[i].Phase == decision.Phase && out[i].Family == decision.Family {
				if out[i].UpdatedAt.Before(decision.UpdatedAt) || out[i].UpdatedAt.IsZero() {
					out[i] = decision
				}
				replaced = true
				break
			}
		}
		if !replaced {
			out = append(out, decision)
		}
	}
	return out
}

func mergeInnerLoop(existing, incoming InnerLoopState) InnerLoopState {
	merged := incoming
	merged.Status = valueOrFallback(merged.Status, existing.Status)
	merged.SearchMode = valueOrFallback(merged.SearchMode, existing.SearchMode)
	merged.BeamWidth = nonZeroInt(merged.BeamWidth, existing.BeamWidth)
	merged.CurrentRound = nonZeroInt(merged.CurrentRound, existing.CurrentRound)
	merged.BestCandidateID = valueOrFallback(merged.BestCandidateID, existing.BestCandidateID)
	merged.StartedAt = chooseCreatedAt(existing.StartedAt, merged.StartedAt)
	merged.UpdatedAt = laterTime(existing.UpdatedAt, merged.UpdatedAt)
	return merged
}
