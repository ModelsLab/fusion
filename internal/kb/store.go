package kb

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	embedded "github.com/ModelsLab/fusion/knowledge"
)

type searchEngine interface {
	Search(query, kind string, limit int) []SearchHit
}

type Store struct {
	Sources    []Source
	GPUs       []GPUProfile
	Strategies []Strategy
	Skills     []Skill
	Examples   []Example
	Documents  []Document

	search searchEngine
}

func LoadDefault() (*Store, error) {
	configDir, err := os.UserConfigDir()
	if err == nil {
		userKnowledgePath := filepath.Join(configDir, "fusion", "knowledge", "knowledge.db")
		if _, statErr := os.Stat(userKnowledgePath); statErr == nil {
			store, loadErr := LoadFromSQLitePath(userKnowledgePath)
			if loadErr == nil {
				return store, nil
			}
		}
	}
	return LoadFromFS(embedded.Files)
}

func LoadFromFS(files fs.FS) (*Store, error) {
	index, err := newSQLiteSearchIndexFromFS(files)
	if err != nil {
		return nil, err
	}
	sqliteIndex, ok := index.(*sqliteSearchIndex)
	if !ok {
		return nil, fmt.Errorf("embedded knowledge store did not return a sqlite index")
	}
	store, err := loadStoreFromSQLite(sqliteIndex.db)
	if err != nil {
		_ = sqliteIndex.db.Close()
		return nil, err
	}
	store.search = sqliteIndex
	return store, nil
}

func (s *Store) SourceByID(id string) (Source, bool) {
	target := canonical(id)
	for _, source := range s.Sources {
		if canonical(source.ID) == target {
			return source, true
		}
	}
	return Source{}, false
}

func (s *Store) GPUByID(id string) (*GPUProfile, bool) {
	target := canonical(id)
	for i := range s.GPUs {
		gpu := &s.GPUs[i]
		if canonical(gpu.ID) == target || canonical(gpu.Name) == target {
			return gpu, true
		}
		for _, alias := range gpu.Aliases {
			if canonical(alias) == target {
				return gpu, true
			}
		}
	}
	return nil, false
}

func (s *Store) StrategyByID(id string) (Strategy, bool) {
	target := canonical(id)
	for _, strategy := range s.Strategies {
		if canonical(strategy.ID) == target {
			return strategy, true
		}
	}
	return Strategy{}, false
}

func (s *Store) SkillByID(id string) (Skill, bool) {
	target := canonical(id)
	for _, skill := range s.Skills {
		if canonical(skill.ID) == target {
			return skill, true
		}
	}
	return Skill{}, false
}

func (s *Store) ExampleByID(id string) (Example, bool) {
	target := canonical(id)
	for _, example := range s.Examples {
		if canonical(example.ID) == target {
			return example, true
		}
	}
	return Example{}, false
}

func (s *Store) DocumentByID(id string) (Document, bool) {
	target := canonical(id)
	for _, document := range s.Documents {
		if canonical(document.ID) == target {
			return document, true
		}
	}
	return Document{}, false
}

func (s *Store) Search(query, kind string, limit int) []SearchHit {
	if s.search != nil {
		return s.search.Search(query, kind, limit)
	}

	tokens := tokenize(query)
	hits := []SearchHit{}

	switch normalizeKind(kind) {
	case "source":
		hits = append(hits, s.searchSources(tokens)...)
	case "gpu":
		hits = append(hits, s.searchGPUs(tokens)...)
	case "strategy":
		hits = append(hits, s.searchStrategies(tokens)...)
	case "skill":
		hits = append(hits, s.searchSkills(tokens)...)
	case "example":
		hits = append(hits, s.searchExamples(tokens)...)
	case "document":
		hits = append(hits, s.searchDocuments(tokens)...)
	default:
		hits = append(hits, s.searchSources(tokens)...)
		hits = append(hits, s.searchGPUs(tokens)...)
		hits = append(hits, s.searchStrategies(tokens)...)
		hits = append(hits, s.searchSkills(tokens)...)
		hits = append(hits, s.searchExamples(tokens)...)
		hits = append(hits, s.searchDocuments(tokens)...)
	}

	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].Title < hits[j].Title
		}
		return hits[i].Score > hits[j].Score
	})

	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}

	return hits
}

func (s *Store) SourcesForIDs(ids []string) []Source {
	out := make([]Source, 0, len(ids))
	seen := map[string]struct{}{}
	for _, id := range ids {
		source, ok := s.SourceByID(id)
		if !ok {
			continue
		}
		if _, exists := seen[source.ID]; exists {
			continue
		}
		seen[source.ID] = struct{}{}
		out = append(out, source)
	}

	sort.Slice(out, func(i, j int) bool {
		if sourcePriority(out[i]) == sourcePriority(out[j]) {
			return out[i].Title < out[j].Title
		}
		return sourcePriority(out[i]) < sourcePriority(out[j])
	})

	return out
}

func (s *Store) IndexRecords() ([]IndexRecord, error) {
	records := make([]IndexRecord, 0, len(s.Sources)+len(s.GPUs)+len(s.Strategies)+len(s.Skills)+len(s.Examples)+len(s.Documents))

	for _, source := range s.Sources {
		payload, err := json.Marshal(source)
		if err != nil {
			return nil, fmt.Errorf("marshal source %s: %w", source.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:         "source",
			ID:           source.ID,
			Title:        source.Title,
			Summary:      source.Summary,
			Body:         strings.Join([]string{source.Title, source.Category, source.Summary, strings.Join(source.Tags, " "), source.URL}, " "),
			Category:     source.Category,
			Reliability:  source.Reliability,
			ReviewStatus: source.ReviewStatus,
			JSON:         string(payload),
		})
	}

	for _, gpu := range s.GPUs {
		payload, err := json.Marshal(gpu)
		if err != nil {
			return nil, fmt.Errorf("marshal gpu %s: %w", gpu.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:        "gpu",
			ID:          gpu.ID,
			Title:       gpu.Name,
			Summary:     strings.Join(gpu.Strengths, "; "),
			Body:        strings.Join([]string{gpu.Name, gpu.Family, gpu.Market, gpu.ComputeCapability, strings.Join(gpu.Aliases, " "), strings.Join(gpu.PreferredPrecisions, " "), strings.Join(gpu.ExperimentalPrecisons, " "), strings.Join(gpu.Strengths, " "), strings.Join(gpu.Constraints, " ")}, " "),
			Category:    gpu.Family,
			Reliability: "curated",
			JSON:        string(payload),
		})
	}

	for _, strategy := range s.Strategies {
		payload, err := json.Marshal(strategy)
		if err != nil {
			return nil, fmt.Errorf("marshal strategy %s: %w", strategy.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:         "strategy",
			ID:           strategy.ID,
			Title:        strategy.Title,
			Summary:      strategy.Summary,
			Body:         strings.Join([]string{strategy.Title, strategy.Category, strategy.Summary, strings.Join(strategy.Workloads, " "), strings.Join(strategy.Operators, " "), strings.Join(strategy.GPUFamilies, " "), strings.Join(strategy.GPUIDs, " "), strings.Join(strategy.Precision, " "), strings.Join(strategy.Bottlenecks, " "), strings.Join(strategy.Goals, " "), strings.Join(strategy.Actions, " "), strings.Join(strategy.Tradeoffs, " ")}, " "),
			Category:     strategy.Category,
			SupportLevel: strategy.SupportLevel,
			Reliability:  "curated",
			ReviewStatus: "reviewed",
			JSON:         string(payload),
		})
	}

	for _, skill := range s.Skills {
		payload, err := json.Marshal(skill)
		if err != nil {
			return nil, fmt.Errorf("marshal skill %s: %w", skill.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:         "skill",
			ID:           skill.ID,
			Title:        skill.Title,
			Summary:      skill.Summary,
			Body:         strings.Join([]string{skill.Title, skill.Category, skill.Summary, strings.Join(skill.Triggers.GPUFamilies, " "), strings.Join(skill.Triggers.GPUIDs, " "), strings.Join(skill.Triggers.Workloads, " "), strings.Join(skill.Triggers.Operators, " "), strings.Join(skill.Triggers.Precision, " "), strings.Join(skill.Triggers.Bottlenecks, " "), strings.Join(skill.Triggers.Runtimes, " "), strings.Join(skill.Triggers.Goals, " "), strings.Join(skill.PreferredBackends, " "), strings.Join(skill.RequiredTools, " "), strings.Join(skill.Steps, " "), strings.Join(skill.Verification, " "), strings.Join(skill.BenchmarkRubric, " "), strings.Join(skill.RuntimeAdapters, " ")}, " "),
			Category:     skill.Category,
			SupportLevel: skill.SupportLevel,
			Reliability:  "curated",
			ReviewStatus: "reviewed",
			JSON:         string(payload),
		})
	}

	for _, example := range s.Examples {
		payload, err := json.Marshal(example)
		if err != nil {
			return nil, fmt.Errorf("marshal example %s: %w", example.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:         "example",
			ID:           example.ID,
			Title:        example.Title,
			Summary:      example.Summary,
			Body:         strings.Join([]string{example.Title, example.Category, example.Backend, example.Summary, strings.Join(example.GPUFamilies, " "), strings.Join(example.GPUIDs, " "), strings.Join(example.Workloads, " "), strings.Join(example.Operators, " "), strings.Join(example.Precision, " "), strings.Join(example.Runtimes, " "), strings.Join(example.UseCases, " "), strings.Join(example.Notes, " "), strings.Join(example.ReferencePaths, " ")}, " "),
			Category:     example.Category,
			SupportLevel: example.SupportLevel,
			Reliability:  "curated",
			ReviewStatus: "reviewed",
			JSON:         string(payload),
		})
	}

	for _, document := range s.Documents {
		payload, err := json.Marshal(document)
		if err != nil {
			return nil, fmt.Errorf("marshal document %s: %w", document.ID, err)
		}
		records = append(records, IndexRecord{
			Kind:         "document",
			ID:           document.ID,
			Title:        document.Title,
			Summary:      document.Summary,
			Body:         strings.Join([]string{document.Title, document.Category, document.Summary, strings.Join(document.Tags, " "), strings.Join(document.GPUFamilies, " "), strings.Join(document.GPUIDs, " "), strings.Join(document.Workloads, " "), strings.Join(document.Operators, " "), strings.Join(document.Precision, " "), strings.Join(document.Runtimes, " "), strings.Join(document.Backends, " "), document.URL, document.Path, document.Body}, " "),
			Category:     document.Category,
			SupportLevel: document.SupportLevel,
			Reliability:  document.Reliability,
			ReviewStatus: document.ReviewStatus,
			JSON:         string(payload),
		})
	}

	return records, nil
}

func (s *Store) searchSources(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, source := range s.Sources {
		score := scoreText(tokens, source.ID, source.Title, source.Category, source.Summary, strings.Join(source.Tags, " ")) + sourceSearchBonus(source)
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "source",
			ID:      source.ID,
			Title:   source.Title,
			Summary: source.Summary,
			Score:   score,
		})
	}
	return hits
}

func (s *Store) searchGPUs(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, gpu := range s.GPUs {
		score := scoreText(tokens, gpu.ID, gpu.Name, gpu.Family, gpu.Market, strings.Join(gpu.Aliases, " "), strings.Join(gpu.Strengths, " "))
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "gpu",
			ID:      gpu.ID,
			Title:   gpu.Name,
			Summary: strings.Join(gpu.Strengths, "; "),
			Score:   score,
		})
	}
	return hits
}

func (s *Store) searchStrategies(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, strategy := range s.Strategies {
		score := scoreText(tokens, strategy.ID, strategy.Title, strategy.Category, strategy.Summary, strings.Join(strategy.Workloads, " "), strings.Join(strategy.Operators, " "), strings.Join(strategy.Goals, " "))
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "strategy",
			ID:      strategy.ID,
			Title:   strategy.Title,
			Summary: strategy.Summary,
			Score:   score,
		})
	}
	return hits
}

func (s *Store) searchSkills(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, skill := range s.Skills {
		score := scoreText(tokens, skill.ID, skill.Title, skill.Category, skill.Summary, strings.Join(skill.Triggers.GPUFamilies, " "), strings.Join(skill.Triggers.Workloads, " "), strings.Join(skill.Triggers.Operators, " "), strings.Join(skill.PreferredBackends, " "))
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "skill",
			ID:      skill.ID,
			Title:   skill.Title,
			Summary: skill.Summary,
			Score:   score,
		})
	}
	return hits
}

func (s *Store) searchExamples(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, example := range s.Examples {
		score := scoreText(tokens, example.ID, example.Title, example.Category, example.Backend, example.Summary, strings.Join(example.GPUFamilies, " "), strings.Join(example.Workloads, " "), strings.Join(example.Operators, " "), strings.Join(example.UseCases, " "))
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "example",
			ID:      example.ID,
			Title:   example.Title,
			Summary: example.Summary,
			Score:   score,
		})
	}
	return hits
}

func (s *Store) searchDocuments(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, document := range s.Documents {
		score := scoreText(tokens, document.ID, document.Title, document.Category, document.Summary, document.URL, document.Path, strings.Join(document.Tags, " "), strings.Join(document.GPUFamilies, " "), strings.Join(document.Workloads, " "), strings.Join(document.Operators, " "), strings.Join(document.Backends, " "), document.Body)
		if score == 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Kind:    "document",
			ID:      document.ID,
			Title:   document.Title,
			Summary: document.Summary,
			Score:   score,
		})
	}
	return hits
}

func normalizeKind(kind string) string {
	switch canonical(kind) {
	case "sources", "source":
		return "source"
	case "gpus", "gpu":
		return "gpu"
	case "strategies", "strategy":
		return "strategy"
	case "skills", "skill":
		return "skill"
	case "examples", "example", "pattern", "patterns":
		return "example"
	case "documents", "document", "doc", "docs", "note", "notes", "markdown":
		return "document"
	default:
		return "all"
	}
}

func tokenize(input string) []string {
	return strings.Fields(canonical(input))
}

func scoreText(tokens []string, fields ...string) int {
	if len(tokens) == 0 {
		return 0
	}

	canonicalFields := make([]string, 0, len(fields))
	for _, field := range fields {
		canonicalFields = append(canonicalFields, canonical(field))
	}

	score := 0
	query := strings.Join(tokens, " ")
	for _, field := range canonicalFields {
		if field == "" {
			continue
		}
		if field == query {
			score += 50
		}
		if strings.Contains(field, query) {
			score += 20
		}
		for _, token := range tokens {
			if strings.Contains(field, token) {
				score += 8
			}
		}
	}

	return score
}

func sourcePriority(source Source) int {
	switch canonical(source.Reliability) {
	case "official":
		return 0
	case "paper":
		return 1
	case "repo":
		return 2
	case "article":
		return 3
	case "curated":
		return 4
	default:
		return 5
	}
}

func sourceSearchBonus(source Source) int {
	score := 0
	switch canonical(source.Reliability) {
	case "official":
		score += 12
	case "paper":
		score += 10
	case "repo":
		score += 8
	case "article":
		score += 6
	}
	if canonical(source.ReviewStatus) == "reviewed" {
		score += 4
	}
	return score
}

func canonical(value string) string {
	replacer := strings.NewReplacer("-", "", "_", "", "/", "", ".", "", ",", "", " ", "")
	return strings.ToLower(strings.TrimSpace(replacer.Replace(value)))
}
