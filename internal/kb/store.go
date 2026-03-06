package kb

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	embedded "github.com/ModelsLab/fusion/knowledge"
)

type Store struct {
	Sources    []Source
	GPUs       []GPUProfile
	Strategies []Strategy
}

func LoadDefault() (*Store, error) {
	return LoadFromFS(embedded.Files)
}

func LoadFromFS(files fs.FS) (*Store, error) {
	var sources []Source
	if err := loadJSON(files, "sources.json", &sources); err != nil {
		return nil, err
	}

	var gpus []GPUProfile
	if err := loadJSON(files, "gpus.json", &gpus); err != nil {
		return nil, err
	}

	var strategies []Strategy
	if err := loadJSON(files, "strategies.json", &strategies); err != nil {
		return nil, err
	}

	return &Store{
		Sources:    sources,
		GPUs:       gpus,
		Strategies: strategies,
	}, nil
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

func (s *Store) Search(query, kind string, limit int) []SearchHit {
	tokens := tokenize(query)
	hits := []SearchHit{}

	switch normalizeKind(kind) {
	case "source":
		hits = append(hits, s.searchSources(tokens)...)
	case "gpu":
		hits = append(hits, s.searchGPUs(tokens)...)
	case "strategy":
		hits = append(hits, s.searchStrategies(tokens)...)
	default:
		hits = append(hits, s.searchSources(tokens)...)
		hits = append(hits, s.searchGPUs(tokens)...)
		hits = append(hits, s.searchStrategies(tokens)...)
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
		return out[i].Title < out[j].Title
	})

	return out
}

func (s *Store) searchSources(tokens []string) []SearchHit {
	hits := []SearchHit{}
	for _, source := range s.Sources {
		score := scoreText(tokens, source.ID, source.Title, source.Category, source.Summary, strings.Join(source.Tags, " "))
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
		score := scoreText(tokens,
			strategy.ID,
			strategy.Title,
			strategy.Category,
			strategy.Summary,
			strings.Join(strategy.Workloads, " "),
			strings.Join(strategy.Operators, " "),
			strings.Join(strategy.Goals, " "),
		)
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

func loadJSON(files fs.FS, name string, target any) error {
	data, err := fs.ReadFile(files, name)
	if err != nil {
		return fmt.Errorf("read %s: %w", name, err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("decode %s: %w", name, err)
	}
	return nil
}

func normalizeKind(kind string) string {
	switch canonical(kind) {
	case "sources", "source":
		return "source"
	case "gpus", "gpu":
		return "gpu"
	case "strategies", "strategy":
		return "strategy"
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

func canonical(value string) string {
	replacer := strings.NewReplacer("-", "", "_", "", "/", "", ".", "", ",", "", " ", "")
	return strings.ToLower(strings.TrimSpace(replacer.Replace(value)))
}
