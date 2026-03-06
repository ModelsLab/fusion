package kb

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

type markdownFrontMatter struct {
	ID                    string   `yaml:"id"`
	Kind                  string   `yaml:"kind"`
	Title                 string   `yaml:"title"`
	Type                  string   `yaml:"type"`
	Category              string   `yaml:"category"`
	Summary               string   `yaml:"summary"`
	SupportLevel          string   `yaml:"support_level"`
	Reliability           string   `yaml:"reliability"`
	ReviewStatus          string   `yaml:"review_status"`
	URL                   string   `yaml:"url"`
	Tags                  []string `yaml:"tags"`
	Aliases               []string `yaml:"aliases"`
	Family                string   `yaml:"family"`
	Market                string   `yaml:"market"`
	ComputeCapability     string   `yaml:"compute_capability"`
	MemoryGB              int      `yaml:"memory_gb"`
	MemoryBandwidthGBps   int      `yaml:"memory_bandwidth_gbps"`
	PreferredPrecisions   []string `yaml:"preferred_precisions"`
	ExperimentalPrecisons []string `yaml:"experimental_precisions"`
	Strengths             []string `yaml:"strengths"`
	Constraints           []string `yaml:"constraints"`
	SourceIDs             []string `yaml:"source_ids"`
	Workloads             []string `yaml:"workloads"`
	Operators             []string `yaml:"operators"`
	GPUFamilies           []string `yaml:"gpu_families"`
	GPUIDs                []string `yaml:"gpu_ids"`
	Precision             []string `yaml:"precision"`
	Bottlenecks           []string `yaml:"bottlenecks"`
	Goals                 []string `yaml:"goals"`
	Priority              int      `yaml:"priority"`
	Preconditions         []string `yaml:"preconditions"`
	Actions               []string `yaml:"actions"`
	Metrics               []string `yaml:"metrics"`
	Tradeoffs             []string `yaml:"tradeoffs"`
	PreferredBackends     []string `yaml:"preferred_backends"`
	RequiredTools         []string `yaml:"required_tools"`
	Steps                 []string `yaml:"steps"`
	Verification          []string `yaml:"verification"`
	BenchmarkRubric       []string `yaml:"benchmark_rubric"`
	FailureRecovery       []string `yaml:"failure_recovery"`
	ArtifactsToSave       []string `yaml:"artifacts_to_save"`
	RuntimeAdapters       []string `yaml:"runtime_adapters"`
	ReferenceSourceIDs    []string `yaml:"reference_source_ids"`
	Backend               string   `yaml:"backend"`
	Runtimes              []string `yaml:"runtimes"`
	UseCases              []string `yaml:"use_cases"`
	Notes                 []string `yaml:"notes"`
	ReferencePaths        []string `yaml:"reference_paths"`
	Backends              []string `yaml:"backends"`
	Path                  string   `yaml:"path"`
}

var errMissingFrontMatter = errors.New("missing leading YAML front matter delimiter")

func HasMarkdownFiles(root string) (bool, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return false, nil
	}
	info, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if !info.IsDir() {
		return false, nil
	}
	found := false
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(d.Name(), "README.md") {
			return nil
		}
		if strings.EqualFold(filepath.Ext(d.Name()), ".md") {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found, err
}

func BuildMarkdownBackedStore(root string, fallback *Store) (*Store, error) {
	hasDocs, err := HasMarkdownFiles(root)
	if err != nil {
		return nil, fmt.Errorf("scan markdown knowledge base: %w", err)
	}
	if hasDocs {
		return LoadMarkdownDir(root)
	}
	if fallback == nil {
		return nil, fmt.Errorf("no markdown knowledge found at %s", root)
	}
	if err := ExportMarkdown(fallback, root); err != nil {
		return nil, err
	}
	return LoadMarkdownDir(root)
}

func LoadMarkdownDir(root string) (*Store, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("markdown knowledge root is required")
	}

	store := &Store{
		Sources:    []Source{},
		GPUs:       []GPUProfile{},
		Strategies: []Strategy{},
		Skills:     []Skill{},
		Examples:   []Example{},
		Documents:  []Document{},
	}

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(d.Name()), ".md") || strings.EqualFold(d.Name(), "README.md") {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}
		meta, body, err := parseMarkdownEntry(path, root, data)
		if err != nil {
			return err
		}

		switch normalizeKind(meta.Kind) {
		case "source":
			store.Sources = append(store.Sources, Source{
				ID:           meta.ID,
				Title:        meta.Title,
				URL:          meta.URL,
				Type:         meta.Type,
				Category:     meta.Category,
				Reliability:  meta.Reliability,
				ReviewStatus: meta.ReviewStatus,
				Summary:      meta.Summary,
				Tags:         append([]string{}, meta.Tags...),
			})
		case "gpu":
			store.GPUs = append(store.GPUs, GPUProfile{
				ID:                    meta.ID,
				Name:                  meta.Title,
				Aliases:               append([]string{}, meta.Aliases...),
				Family:                meta.Family,
				Market:                meta.Market,
				ComputeCapability:     meta.ComputeCapability,
				MemoryGB:              meta.MemoryGB,
				MemoryBandwidthGBps:   meta.MemoryBandwidthGBps,
				PreferredPrecisions:   append([]string{}, meta.PreferredPrecisions...),
				ExperimentalPrecisons: append([]string{}, meta.ExperimentalPrecisons...),
				Strengths:             append([]string{}, meta.Strengths...),
				Constraints:           append([]string{}, meta.Constraints...),
				SourceIDs:             append([]string{}, meta.SourceIDs...),
			})
		case "strategy":
			store.Strategies = append(store.Strategies, Strategy{
				ID:            meta.ID,
				Title:         meta.Title,
				Category:      meta.Category,
				Summary:       meta.Summary,
				SupportLevel:  meta.SupportLevel,
				Workloads:     append([]string{}, meta.Workloads...),
				Operators:     append([]string{}, meta.Operators...),
				GPUFamilies:   append([]string{}, meta.GPUFamilies...),
				GPUIDs:        append([]string{}, meta.GPUIDs...),
				Precision:     append([]string{}, meta.Precision...),
				Bottlenecks:   append([]string{}, meta.Bottlenecks...),
				Goals:         append([]string{}, meta.Goals...),
				Priority:      meta.Priority,
				Preconditions: append([]string{}, meta.Preconditions...),
				Actions:       append([]string{}, meta.Actions...),
				Metrics:       append([]string{}, meta.Metrics...),
				Tradeoffs:     append([]string{}, meta.Tradeoffs...),
				SourceIDs:     append([]string{}, meta.SourceIDs...),
			})
		case "skill":
			store.Skills = append(store.Skills, Skill{
				ID:           meta.ID,
				Title:        meta.Title,
				Category:     meta.Category,
				Summary:      meta.Summary,
				SupportLevel: meta.SupportLevel,
				Triggers: SkillTrigger{
					GPUFamilies: append([]string{}, meta.GPUFamilies...),
					GPUIDs:      append([]string{}, meta.GPUIDs...),
					Workloads:   append([]string{}, meta.Workloads...),
					Operators:   append([]string{}, meta.Operators...),
					Precision:   append([]string{}, meta.Precision...),
					Bottlenecks: append([]string{}, meta.Bottlenecks...),
					Runtimes:    append([]string{}, meta.Runtimes...),
					Goals:       append([]string{}, meta.Goals...),
				},
				PreferredBackends:  append([]string{}, meta.PreferredBackends...),
				RequiredTools:      append([]string{}, meta.RequiredTools...),
				Steps:              append([]string{}, meta.Steps...),
				Verification:       append([]string{}, meta.Verification...),
				BenchmarkRubric:    append([]string{}, meta.BenchmarkRubric...),
				FailureRecovery:    append([]string{}, meta.FailureRecovery...),
				ArtifactsToSave:    append([]string{}, meta.ArtifactsToSave...),
				RuntimeAdapters:    append([]string{}, meta.RuntimeAdapters...),
				ReferenceSourceIDs: append([]string{}, meta.ReferenceSourceIDs...),
			})
		case "example":
			store.Examples = append(store.Examples, Example{
				ID:             meta.ID,
				Title:          meta.Title,
				Category:       meta.Category,
				Backend:        meta.Backend,
				Summary:        meta.Summary,
				SupportLevel:   meta.SupportLevel,
				GPUFamilies:    append([]string{}, meta.GPUFamilies...),
				GPUIDs:         append([]string{}, meta.GPUIDs...),
				Workloads:      append([]string{}, meta.Workloads...),
				Operators:      append([]string{}, meta.Operators...),
				Precision:      append([]string{}, meta.Precision...),
				Runtimes:       append([]string{}, meta.Runtimes...),
				UseCases:       append([]string{}, meta.UseCases...),
				Notes:          append([]string{}, meta.Notes...),
				ReferencePaths: append([]string{}, meta.ReferencePaths...),
				SourceIDs:      append([]string{}, meta.SourceIDs...),
			})
		case "document":
			store.Documents = append(store.Documents, Document{
				ID:           meta.ID,
				Title:        meta.Title,
				Category:     meta.Category,
				Summary:      meta.Summary,
				SupportLevel: meta.SupportLevel,
				Reliability:  meta.Reliability,
				ReviewStatus: meta.ReviewStatus,
				URL:          meta.URL,
				Tags:         append([]string{}, meta.Tags...),
				GPUFamilies:  append([]string{}, meta.GPUFamilies...),
				GPUIDs:       append([]string{}, meta.GPUIDs...),
				Workloads:    append([]string{}, meta.Workloads...),
				Operators:    append([]string{}, meta.Operators...),
				Precision:    append([]string{}, meta.Precision...),
				Runtimes:     append([]string{}, meta.Runtimes...),
				Backends:     append([]string{}, meta.Backends...),
				SourceIDs:    append([]string{}, meta.SourceIDs...),
				Path:         meta.Path,
				Body:         body,
			})
		default:
			return fmt.Errorf("unsupported knowledge kind %q in %s", meta.Kind, path)
		}

		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Slice(store.Sources, func(i, j int) bool { return store.Sources[i].ID < store.Sources[j].ID })
	sort.Slice(store.GPUs, func(i, j int) bool { return store.GPUs[i].ID < store.GPUs[j].ID })
	sort.Slice(store.Strategies, func(i, j int) bool { return store.Strategies[i].ID < store.Strategies[j].ID })
	sort.Slice(store.Skills, func(i, j int) bool { return store.Skills[i].ID < store.Skills[j].ID })
	sort.Slice(store.Examples, func(i, j int) bool { return store.Examples[i].ID < store.Examples[j].ID })
	sort.Slice(store.Documents, func(i, j int) bool { return store.Documents[i].ID < store.Documents[j].ID })

	return store, nil
}

func ExportMarkdown(store *Store, root string) error {
	root = strings.TrimSpace(root)
	if root == "" {
		return fmt.Errorf("markdown knowledge root is required")
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return fmt.Errorf("create markdown knowledge root: %w", err)
	}

	for _, source := range store.Sources {
		body := renderMarkdownSections(
			markdownSection{Title: "Summary", Lines: []string{source.Summary}},
			markdownSection{Title: "Tags", BulletLines: source.Tags},
			markdownSection{Title: "Reference", Lines: []string{source.URL}},
		)
		if err := writeMarkdownEntry(filepath.Join(root, "sources", sanitizeFilename(source.ID)+".md"), markdownFrontMatter{
			ID:           source.ID,
			Kind:         "source",
			Title:        source.Title,
			Type:         source.Type,
			Category:     source.Category,
			Summary:      source.Summary,
			Reliability:  source.Reliability,
			ReviewStatus: source.ReviewStatus,
			URL:          source.URL,
			Tags:         source.Tags,
		}, body); err != nil {
			return err
		}
	}

	for _, gpu := range store.GPUs {
		body := renderMarkdownSections(
			markdownSection{Title: "Strengths", BulletLines: gpu.Strengths},
			markdownSection{Title: "Constraints", BulletLines: gpu.Constraints},
			markdownSection{Title: "Precisions", BulletLines: append(append([]string{}, gpu.PreferredPrecisions...), gpu.ExperimentalPrecisons...)},
		)
		if err := writeMarkdownEntry(filepath.Join(root, "gpus", sanitizeFilename(gpu.ID)+".md"), markdownFrontMatter{
			ID:                    gpu.ID,
			Kind:                  "gpu",
			Title:                 gpu.Name,
			Aliases:               gpu.Aliases,
			Family:                gpu.Family,
			Market:                gpu.Market,
			ComputeCapability:     gpu.ComputeCapability,
			MemoryGB:              gpu.MemoryGB,
			MemoryBandwidthGBps:   gpu.MemoryBandwidthGBps,
			PreferredPrecisions:   gpu.PreferredPrecisions,
			ExperimentalPrecisons: gpu.ExperimentalPrecisons,
			Strengths:             gpu.Strengths,
			Constraints:           gpu.Constraints,
			SourceIDs:             gpu.SourceIDs,
		}, body); err != nil {
			return err
		}
	}

	for _, strategy := range store.Strategies {
		body := renderMarkdownSections(
			markdownSection{Title: "Actions", BulletLines: strategy.Actions},
			markdownSection{Title: "Tradeoffs", BulletLines: strategy.Tradeoffs},
			markdownSection{Title: "Metrics", BulletLines: strategy.Metrics},
		)
		if err := writeMarkdownEntry(filepath.Join(root, "strategies", sanitizeFilename(strategy.ID)+".md"), markdownFrontMatter{
			ID:            strategy.ID,
			Kind:          "strategy",
			Title:         strategy.Title,
			Category:      strategy.Category,
			Summary:       strategy.Summary,
			SupportLevel:  strategy.SupportLevel,
			Workloads:     strategy.Workloads,
			Operators:     strategy.Operators,
			GPUFamilies:   strategy.GPUFamilies,
			GPUIDs:        strategy.GPUIDs,
			Precision:     strategy.Precision,
			Bottlenecks:   strategy.Bottlenecks,
			Goals:         strategy.Goals,
			Priority:      strategy.Priority,
			Preconditions: strategy.Preconditions,
			Actions:       strategy.Actions,
			Metrics:       strategy.Metrics,
			Tradeoffs:     strategy.Tradeoffs,
			SourceIDs:     strategy.SourceIDs,
		}, body); err != nil {
			return err
		}
	}

	for _, skill := range store.Skills {
		body := renderMarkdownSections(
			markdownSection{Title: "Steps", BulletLines: skill.Steps},
			markdownSection{Title: "Verification", BulletLines: skill.Verification},
			markdownSection{Title: "Benchmark Rubric", BulletLines: skill.BenchmarkRubric},
			markdownSection{Title: "Failure Recovery", BulletLines: skill.FailureRecovery},
		)
		if err := writeMarkdownEntry(filepath.Join(root, "skills", sanitizeFilename(skill.ID)+".md"), markdownFrontMatter{
			ID:                 skill.ID,
			Kind:               "skill",
			Title:              skill.Title,
			Category:           skill.Category,
			Summary:            skill.Summary,
			SupportLevel:       skill.SupportLevel,
			GPUFamilies:        skill.Triggers.GPUFamilies,
			GPUIDs:             skill.Triggers.GPUIDs,
			Workloads:          skill.Triggers.Workloads,
			Operators:          skill.Triggers.Operators,
			Precision:          skill.Triggers.Precision,
			Bottlenecks:        skill.Triggers.Bottlenecks,
			Runtimes:           skill.Triggers.Runtimes,
			Goals:              skill.Triggers.Goals,
			PreferredBackends:  skill.PreferredBackends,
			RequiredTools:      skill.RequiredTools,
			Steps:              skill.Steps,
			Verification:       skill.Verification,
			BenchmarkRubric:    skill.BenchmarkRubric,
			FailureRecovery:    skill.FailureRecovery,
			ArtifactsToSave:    skill.ArtifactsToSave,
			RuntimeAdapters:    skill.RuntimeAdapters,
			ReferenceSourceIDs: skill.ReferenceSourceIDs,
		}, body); err != nil {
			return err
		}
	}

	for _, example := range store.Examples {
		body := renderMarkdownSections(
			markdownSection{Title: "Use Cases", BulletLines: example.UseCases},
			markdownSection{Title: "Notes", BulletLines: example.Notes},
			markdownSection{Title: "Reference Paths", BulletLines: example.ReferencePaths},
		)
		if err := writeMarkdownEntry(filepath.Join(root, "examples", sanitizeFilename(example.ID)+".md"), markdownFrontMatter{
			ID:             example.ID,
			Kind:           "example",
			Title:          example.Title,
			Category:       example.Category,
			Backend:        example.Backend,
			Summary:        example.Summary,
			SupportLevel:   example.SupportLevel,
			GPUFamilies:    example.GPUFamilies,
			GPUIDs:         example.GPUIDs,
			Workloads:      example.Workloads,
			Operators:      example.Operators,
			Precision:      example.Precision,
			Runtimes:       example.Runtimes,
			UseCases:       example.UseCases,
			Notes:          example.Notes,
			ReferencePaths: example.ReferencePaths,
			SourceIDs:      example.SourceIDs,
		}, body); err != nil {
			return err
		}
	}

	for _, document := range store.Documents {
		body := strings.TrimSpace(document.Body)
		if body == "" {
			body = renderMarkdownSections(markdownSection{Title: "Summary", Lines: []string{document.Summary}})
		}
		targetPath := document.Path
		if strings.TrimSpace(targetPath) == "" {
			targetPath = filepath.Join("documents", sanitizeFilename(document.ID)+".md")
		}
		if err := writeMarkdownEntry(filepath.Join(root, targetPath), markdownFrontMatter{
			ID:           document.ID,
			Kind:         "document",
			Title:        document.Title,
			Category:     document.Category,
			Summary:      document.Summary,
			SupportLevel: document.SupportLevel,
			Reliability:  document.Reliability,
			ReviewStatus: document.ReviewStatus,
			URL:          document.URL,
			Tags:         document.Tags,
			GPUFamilies:  document.GPUFamilies,
			GPUIDs:       document.GPUIDs,
			Workloads:    document.Workloads,
			Operators:    document.Operators,
			Precision:    document.Precision,
			Runtimes:     document.Runtimes,
			Backends:     document.Backends,
			SourceIDs:    document.SourceIDs,
			Path:         filepath.ToSlash(strings.TrimSpace(targetPath)),
		}, body); err != nil {
			return err
		}
	}

	return nil
}

type markdownSection struct {
	Title       string
	Lines       []string
	BulletLines []string
}

func parseMarkdownEntry(path, root string, data []byte) (markdownFrontMatter, string, error) {
	var meta markdownFrontMatter
	frontMatter, body, err := splitFrontMatter(data)
	if err != nil {
		relPath, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return meta, "", fmt.Errorf("resolve relative path %s: %w", path, relErr)
		}
		relPath = filepath.ToSlash(relPath)
		if errors.Is(err, errMissingFrontMatter) {
			meta.Kind = inferKindFromPath(relPath)
			meta.Path = relPath
			meta.ID = strings.TrimSuffix(filepath.Base(relPath), filepath.Ext(relPath))
			meta.Title = humanizeID(meta.ID)
			bodyText := strings.TrimSpace(string(data))
			meta.Summary = deriveSummary(bodyText)
			return meta, bodyText, nil
		}
		return meta, "", fmt.Errorf("parse front matter %s: %w", path, err)
	}
	if err := yaml.Unmarshal(frontMatter, &meta); err != nil {
		return meta, "", fmt.Errorf("decode front matter %s: %w", path, err)
	}

	relPath, err := filepath.Rel(root, path)
	if err != nil {
		return meta, "", fmt.Errorf("resolve relative path %s: %w", path, err)
	}
	relPath = filepath.ToSlash(relPath)
	if strings.TrimSpace(meta.Path) == "" {
		meta.Path = relPath
	}

	if strings.TrimSpace(meta.Kind) == "" {
		meta.Kind = inferKindFromPath(relPath)
	}
	if strings.TrimSpace(meta.ID) == "" {
		meta.ID = strings.TrimSuffix(filepath.Base(relPath), filepath.Ext(relPath))
	}
	if strings.TrimSpace(meta.Title) == "" {
		meta.Title = humanizeID(meta.ID)
	}
	bodyText := strings.TrimSpace(string(body))
	if strings.TrimSpace(meta.Summary) == "" {
		meta.Summary = deriveSummary(bodyText)
	}
	return meta, bodyText, nil
}

func splitFrontMatter(data []byte) ([]byte, []byte, error) {
	text := string(data)
	if !strings.HasPrefix(text, "---\n") && !strings.HasPrefix(text, "---\r\n") {
		return nil, data, errMissingFrontMatter
	}
	trimmed := strings.TrimPrefix(strings.TrimPrefix(text, "---\r\n"), "---\n")
	idx := strings.Index(trimmed, "\n---")
	if idx < 0 {
		return nil, data, fmt.Errorf("missing closing YAML front matter delimiter")
	}
	frontMatter := trimmed[:idx]
	body := strings.TrimPrefix(trimmed[idx+1:], "---")
	body = strings.TrimPrefix(body, "\r\n")
	body = strings.TrimPrefix(body, "\n")
	return []byte(frontMatter), []byte(body), nil
}

func inferKindFromPath(path string) string {
	parts := strings.Split(filepath.ToSlash(path), "/")
	if len(parts) == 0 {
		return "document"
	}
	return normalizeKind(parts[0])
}

func deriveSummary(body string) string {
	body = strings.TrimSpace(body)
	if body == "" {
		return ""
	}
	paragraphs := strings.Split(body, "\n\n")
	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if paragraph == "" || strings.HasPrefix(paragraph, "#") || strings.HasPrefix(paragraph, "- ") {
			continue
		}
		return strings.TrimSpace(paragraph)
	}
	return body
}

func writeMarkdownEntry(path string, meta markdownFrontMatter, body string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create markdown directory: %w", err)
	}
	var frontMatter bytes.Buffer
	encoder := yaml.NewEncoder(&frontMatter)
	encoder.SetIndent(2)
	if err := encoder.Encode(meta); err != nil {
		return fmt.Errorf("encode markdown front matter: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return fmt.Errorf("close markdown front matter encoder: %w", err)
	}
	content := "---\n" + strings.TrimSpace(frontMatter.String()) + "\n---\n\n" + strings.TrimSpace(body) + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("write markdown entry: %w", err)
	}
	return nil
}

func renderMarkdownSections(sections ...markdownSection) string {
	lines := []string{}
	for _, section := range sections {
		if len(section.Lines) == 0 && len(section.BulletLines) == 0 {
			continue
		}
		lines = append(lines, "## "+section.Title, "")
		for _, line := range section.Lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			lines = append(lines, line)
		}
		for _, line := range section.BulletLines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			lines = append(lines, "- "+line)
		}
		lines = append(lines, "")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func sanitizeFilename(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	value = strings.NewReplacer(" ", "-", "/", "-", "_", "-", ".", "-").Replace(value)
	value = strings.Trim(value, "-")
	if value == "" {
		return "entry"
	}
	return value
}

func humanizeID(value string) string {
	value = strings.ReplaceAll(value, "-", " ")
	value = strings.ReplaceAll(value, "_", " ")
	return strings.Title(strings.TrimSpace(value))
}
