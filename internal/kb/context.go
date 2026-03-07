package kb

import (
	"fmt"
	"sort"
	"strings"
)

func (s *Store) BuildContextPacket(input ContextRequest) ContextPacket {
	req := normalizeContextRequest(input)
	packet := ContextPacket{
		Request:    req,
		Strategies: []StrategyMatch{},
		Skills:     []SkillMatch{},
		Examples:   []ExampleMatch{},
		Documents:  []DocumentMatch{},
		Sources:    []Source{},
		Notes:      []string{},
	}

	var gpu *GPUProfile
	if req.GPU != "" {
		if resolved, ok := s.GPUByID(req.GPU); ok {
			gpu = resolved
			packet.GPU = resolved
		} else {
			packet.Notes = append(packet.Notes, fmt.Sprintf("GPU %q is not yet normalized in the curated knowledge base, so retrieval is falling back to family-agnostic matches.", req.GPU))
		}
	}

	bottleneck, bottleneckNote := inferContextBottleneck(req, gpu)
	if req.Bottleneck == "" && bottleneckNote != "" {
		packet.Notes = append(packet.Notes, bottleneckNote)
	}

	for _, strategy := range s.Strategies {
		match, ok := scoreContextStrategy(strategy, req, gpu, bottleneck, s)
		if ok {
			packet.Strategies = append(packet.Strategies, match)
		}
	}
	sort.Slice(packet.Strategies, func(i, j int) bool {
		if packet.Strategies[i].Score == packet.Strategies[j].Score {
			return packet.Strategies[i].Strategy.Title < packet.Strategies[j].Strategy.Title
		}
		return packet.Strategies[i].Score > packet.Strategies[j].Score
	})

	for _, skill := range s.Skills {
		match, ok := scoreContextSkill(skill, req, gpu, bottleneck, s)
		if ok {
			packet.Skills = append(packet.Skills, match)
		}
	}
	sort.Slice(packet.Skills, func(i, j int) bool {
		if packet.Skills[i].Score == packet.Skills[j].Score {
			return packet.Skills[i].Skill.Title < packet.Skills[j].Skill.Title
		}
		return packet.Skills[i].Score > packet.Skills[j].Score
	})

	for _, example := range s.Examples {
		match, ok := scoreContextExample(example, req, gpu, bottleneck, s)
		if ok {
			packet.Examples = append(packet.Examples, match)
		}
	}
	sort.Slice(packet.Examples, func(i, j int) bool {
		if packet.Examples[i].Score == packet.Examples[j].Score {
			return packet.Examples[i].Example.Title < packet.Examples[j].Example.Title
		}
		return packet.Examples[i].Score > packet.Examples[j].Score
	})

	limit := req.Limit
	if limit <= 0 {
		limit = 4
	}
	if len(packet.Strategies) > limit {
		packet.Strategies = packet.Strategies[:limit]
	}
	if len(packet.Skills) > limit {
		packet.Skills = packet.Skills[:limit]
	}
	if len(packet.Examples) > limit {
		packet.Examples = packet.Examples[:limit]
	}
	for _, document := range s.Documents {
		match, ok := scoreContextDocument(document, req, gpu, bottleneck, s)
		if ok {
			packet.Documents = append(packet.Documents, match)
		}
	}
	sort.Slice(packet.Documents, func(i, j int) bool {
		if packet.Documents[i].Score == packet.Documents[j].Score {
			return packet.Documents[i].Document.Title < packet.Documents[j].Document.Title
		}
		return packet.Documents[i].Score > packet.Documents[j].Score
	})
	if len(packet.Documents) > limit {
		packet.Documents = packet.Documents[:limit]
	}

	sourceIDs := []string{}
	for _, match := range packet.Strategies {
		sourceIDs = append(sourceIDs, match.Strategy.SourceIDs...)
	}
	for _, match := range packet.Skills {
		sourceIDs = append(sourceIDs, match.Skill.ReferenceSourceIDs...)
	}
	for _, match := range packet.Examples {
		sourceIDs = append(sourceIDs, match.Example.SourceIDs...)
	}
	for _, match := range packet.Documents {
		sourceIDs = append(sourceIDs, match.Document.SourceIDs...)
	}
	if req.Query != "" {
		for _, hit := range s.Search(req.Query, "source", limit*2) {
			sourceIDs = append(sourceIDs, hit.ID)
		}
	}
	packet.Sources = s.SourcesForIDs(sourceIDs)

	if len(packet.Sources) == 0 {
		packet.Notes = append(packet.Notes, "No directly linked sources were matched, so the agent should widen search with search_knowledge_base.")
	}

	return packet
}

func scoreContextStrategy(strategy Strategy, req ContextRequest, gpu *GPUProfile, bottleneck string, store *Store) (StrategyMatch, bool) {
	if !req.IncludeExperimental && canonical(strategy.SupportLevel) == "experimental" {
		return StrategyMatch{}, false
	}

	score := strategy.Priority
	reasons := []string{}

	if gpu != nil {
		if len(strategy.GPUIDs) > 0 || len(strategy.GPUFamilies) > 0 {
			switch {
			case matchesContext(gpu.ID, strategy.GPUIDs):
				score += 24
				reasons = append(reasons, fmt.Sprintf("directly tuned for %s", gpu.Name))
			case matchesContext(gpu.Family, strategy.GPUFamilies):
				score += 16
				reasons = append(reasons, fmt.Sprintf("matches the %s GPU family", gpu.Family))
			default:
				return StrategyMatch{}, false
			}
		}
	}

	if !matchesOptionalBucket(req.Workload, strategy.Workloads, strategy.Category == "workflow") {
		return StrategyMatch{}, false
	}
	if req.Workload != "" && matchesContext(req.Workload, strategy.Workloads) {
		score += 14
		reasons = append(reasons, fmt.Sprintf("targets %s workloads", req.Workload))
	}

	operatorOverlap := contextOverlap(req.Operators, strategy.Operators)
	if len(req.Operators) > 0 && len(strategy.Operators) > 0 && !containsWildcardValues(strategy.Operators) && len(operatorOverlap) == 0 && strategy.Category != "workflow" {
		return StrategyMatch{}, false
	}
	if len(operatorOverlap) > 0 {
		score += len(operatorOverlap) * 9
		reasons = append(reasons, fmt.Sprintf("covers operators: %s", strings.Join(operatorOverlap, ", ")))
	}

	if !matchesOptionalBucket(req.Precision, strategy.Precision, strategy.Category == "workflow") {
		return StrategyMatch{}, false
	}
	if req.Precision != "" && matchesContext(req.Precision, strategy.Precision) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("applies to %s precision paths", req.Precision))
	}

	if bottleneck != "" && matchesContext(bottleneck, strategy.Bottlenecks) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("fits a %s bottleneck", bottleneck))
	}

	goalOverlap := contextOverlap(req.Goals, strategy.Goals)
	if len(goalOverlap) > 0 {
		score += len(goalOverlap) * 6
		reasons = append(reasons, fmt.Sprintf("advances goals: %s", strings.Join(goalOverlap, ", ")))
	}
	if taskScore := taskQueryScore(req.Task, strategy.ID, strategy.Title, strategy.Summary, strings.Join(strategy.Actions, " "), strings.Join(strategy.Tradeoffs, " ")); taskScore > 0 {
		score += taskScore
		reasons = append(reasons, fmt.Sprintf("relevant to %s tasks", req.Task))
	}

	if queryScore := scoreText(tokenize(req.Query), strategy.ID, strategy.Title, strategy.Summary, strings.Join(strategy.Actions, " "), strings.Join(strategy.Tradeoffs, " ")); queryScore > 0 {
		score += queryScore
		reasons = append(reasons, "directly matched the query text")
	}

	if len(reasons) == 0 {
		reasons = append(reasons, strategy.Summary)
	}

	return StrategyMatch{
		Strategy: strategy,
		Score:    score,
		Reasons:  dedupeStrings(reasons),
		Sources:  store.SourcesForIDs(strategy.SourceIDs),
	}, true
}

func scoreContextSkill(skill Skill, req ContextRequest, gpu *GPUProfile, bottleneck string, store *Store) (SkillMatch, bool) {
	if !req.IncludeExperimental && canonical(skill.SupportLevel) == "experimental" {
		return SkillMatch{}, false
	}

	score := 60
	reasons := []string{}

	if gpu != nil {
		if len(skill.Triggers.GPUIDs) > 0 || len(skill.Triggers.GPUFamilies) > 0 {
			switch {
			case matchesContext(gpu.ID, skill.Triggers.GPUIDs):
				score += 24
				reasons = append(reasons, fmt.Sprintf("skill is directly scoped to %s", gpu.Name))
			case matchesContext(gpu.Family, skill.Triggers.GPUFamilies):
				score += 16
				reasons = append(reasons, fmt.Sprintf("skill is relevant for %s GPUs", gpu.Family))
			default:
				return SkillMatch{}, false
			}
		}
	}

	if !matchesOptionalBucket(req.Workload, skill.Triggers.Workloads, false) {
		return SkillMatch{}, false
	}
	if req.Workload != "" && matchesContext(req.Workload, skill.Triggers.Workloads) {
		score += 12
		reasons = append(reasons, fmt.Sprintf("contains a %s playbook", req.Workload))
	}

	operatorOverlap := contextOverlap(req.Operators, skill.Triggers.Operators)
	if len(req.Operators) > 0 && len(skill.Triggers.Operators) > 0 && !containsWildcardValues(skill.Triggers.Operators) && len(operatorOverlap) == 0 {
		return SkillMatch{}, false
	}
	if len(operatorOverlap) > 0 {
		score += len(operatorOverlap) * 8
		reasons = append(reasons, fmt.Sprintf("matches operators: %s", strings.Join(operatorOverlap, ", ")))
	}

	if !matchesOptionalBucket(req.Precision, skill.Triggers.Precision, false) {
		return SkillMatch{}, false
	}
	if req.Precision != "" && matchesContext(req.Precision, skill.Triggers.Precision) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("matches %s precision work", req.Precision))
	}

	if !matchesOptionalBucket(req.Runtime, skill.Triggers.Runtimes, true) {
		return SkillMatch{}, false
	}
	if req.Runtime != "" && matchesContext(req.Runtime, skill.Triggers.Runtimes) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("includes runtime guidance for %s", req.Runtime))
	}

	if bottleneck != "" && matchesContext(bottleneck, skill.Triggers.Bottlenecks) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("built for %s-bound optimization loops", bottleneck))
	}

	goalOverlap := contextOverlap(req.Goals, skill.Triggers.Goals)
	if len(goalOverlap) > 0 {
		score += len(goalOverlap) * 5
		reasons = append(reasons, fmt.Sprintf("supports goals: %s", strings.Join(goalOverlap, ", ")))
	}
	if taskScore := taskQueryScore(req.Task, skill.ID, skill.Title, skill.Summary, strings.Join(skill.PreferredBackends, " "), strings.Join(skill.Steps, " ")); taskScore > 0 {
		score += taskScore
		reasons = append(reasons, fmt.Sprintf("skill is relevant to %s tasks", req.Task))
	}

	if queryScore := scoreText(tokenize(req.Query), skill.ID, skill.Title, skill.Summary, strings.Join(skill.PreferredBackends, " "), strings.Join(skill.Steps, " ")); queryScore > 0 {
		score += queryScore
		reasons = append(reasons, "playbook text matches the request")
	}

	if len(reasons) == 0 {
		reasons = append(reasons, skill.Summary)
	}

	return SkillMatch{
		Skill:   skill,
		Score:   score,
		Reasons: dedupeStrings(reasons),
		Sources: store.SourcesForIDs(skill.ReferenceSourceIDs),
	}, true
}

func scoreContextExample(example Example, req ContextRequest, gpu *GPUProfile, bottleneck string, store *Store) (ExampleMatch, bool) {
	if !req.IncludeExperimental && canonical(example.SupportLevel) == "experimental" {
		return ExampleMatch{}, false
	}

	score := 48
	reasons := []string{}

	if gpu != nil {
		if len(example.GPUIDs) > 0 || len(example.GPUFamilies) > 0 {
			switch {
			case matchesContext(gpu.ID, example.GPUIDs):
				score += 18
				reasons = append(reasons, fmt.Sprintf("example is specific to %s", gpu.Name))
			case matchesContext(gpu.Family, example.GPUFamilies):
				score += 12
				reasons = append(reasons, fmt.Sprintf("example matches %s GPUs", gpu.Family))
			default:
				return ExampleMatch{}, false
			}
		}
	}

	if !matchesOptionalBucket(req.Workload, example.Workloads, true) {
		return ExampleMatch{}, false
	}
	if req.Workload != "" && matchesContext(req.Workload, example.Workloads) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("example is useful for %s", req.Workload))
	}

	operatorOverlap := contextOverlap(req.Operators, example.Operators)
	if len(req.Operators) > 0 && len(example.Operators) > 0 && !containsWildcardValues(example.Operators) && len(operatorOverlap) == 0 {
		return ExampleMatch{}, false
	}
	if len(operatorOverlap) > 0 {
		score += len(operatorOverlap) * 7
		reasons = append(reasons, fmt.Sprintf("covers operators: %s", strings.Join(operatorOverlap, ", ")))
	}

	if !matchesOptionalBucket(req.Precision, example.Precision, true) {
		return ExampleMatch{}, false
	}
	if req.Precision != "" && matchesContext(req.Precision, example.Precision) {
		score += 8
		reasons = append(reasons, fmt.Sprintf("example aligns with %s precision", req.Precision))
	}

	if !matchesOptionalBucket(req.Runtime, example.Runtimes, true) {
		return ExampleMatch{}, false
	}
	if req.Runtime != "" && matchesContext(req.Runtime, example.Runtimes) {
		score += 8
		reasons = append(reasons, fmt.Sprintf("example mentions %s runtime integration", req.Runtime))
	}

	if bottleneck != "" && containsAnyValue(example.UseCases, bottleneck) {
		score += 6
		reasons = append(reasons, fmt.Sprintf("example is relevant to %s bottlenecks", bottleneck))
	}
	if taskScore := taskQueryScore(req.Task, example.ID, example.Title, example.Category, example.Backend, example.Summary, strings.Join(example.UseCases, " "), strings.Join(example.Notes, " ")); taskScore > 0 {
		score += taskScore
		reasons = append(reasons, fmt.Sprintf("example is relevant to %s tasks", req.Task))
	}

	if queryScore := scoreText(tokenize(req.Query), example.ID, example.Title, example.Category, example.Backend, example.Summary, strings.Join(example.UseCases, " "), strings.Join(example.Notes, " ")); queryScore > 0 {
		score += queryScore
		reasons = append(reasons, "reference text matches the request")
	}

	if len(reasons) == 0 {
		reasons = append(reasons, example.Summary)
	}

	return ExampleMatch{
		Example: example,
		Score:   score,
		Reasons: dedupeStrings(reasons),
		Sources: store.SourcesForIDs(example.SourceIDs),
	}, true
}

func scoreContextDocument(document Document, req ContextRequest, gpu *GPUProfile, bottleneck string, store *Store) (DocumentMatch, bool) {
	if !req.IncludeExperimental && canonical(document.SupportLevel) == "experimental" {
		return DocumentMatch{}, false
	}

	score := 40
	reasons := []string{}

	if gpu != nil {
		if len(document.GPUIDs) > 0 || len(document.GPUFamilies) > 0 {
			switch {
			case matchesContext(gpu.ID, document.GPUIDs):
				score += 20
				reasons = append(reasons, fmt.Sprintf("document is directly relevant to %s", gpu.Name))
			case matchesContext(gpu.Family, document.GPUFamilies):
				score += 14
				reasons = append(reasons, fmt.Sprintf("document applies to %s GPUs", gpu.Family))
			default:
				return DocumentMatch{}, false
			}
		}
	}

	if !matchesOptionalBucket(req.Workload, document.Workloads, true) {
		return DocumentMatch{}, false
	}
	if req.Workload != "" && matchesContext(req.Workload, document.Workloads) {
		score += 10
		reasons = append(reasons, fmt.Sprintf("covers %s workloads", req.Workload))
	}

	operatorOverlap := contextOverlap(req.Operators, document.Operators)
	if len(req.Operators) > 0 && len(document.Operators) > 0 && !containsWildcardValues(document.Operators) && len(operatorOverlap) == 0 {
		return DocumentMatch{}, false
	}
	if len(operatorOverlap) > 0 {
		score += len(operatorOverlap) * 7
		reasons = append(reasons, fmt.Sprintf("matches operators: %s", strings.Join(operatorOverlap, ", ")))
	}

	if !matchesOptionalBucket(req.Precision, document.Precision, true) {
		return DocumentMatch{}, false
	}
	if req.Precision != "" && matchesContext(req.Precision, document.Precision) {
		score += 8
		reasons = append(reasons, fmt.Sprintf("applies to %s precision work", req.Precision))
	}

	if !matchesOptionalBucket(req.Runtime, document.Runtimes, true) {
		return DocumentMatch{}, false
	}
	if req.Runtime != "" && matchesContext(req.Runtime, document.Runtimes) {
		score += 8
		reasons = append(reasons, fmt.Sprintf("discusses %s runtime integration", req.Runtime))
	}

	if bottleneck != "" && scoreText([]string{canonical(bottleneck)}, document.Summary, document.Body) > 0 {
		score += 6
		reasons = append(reasons, fmt.Sprintf("mentions %s bottlenecks", bottleneck))
	}
	if taskScore := taskQueryScore(req.Task, document.ID, document.Title, document.Summary, document.Body); taskScore > 0 {
		score += taskScore
		reasons = append(reasons, fmt.Sprintf("document is relevant to %s tasks", req.Task))
	}

	if queryScore := scoreText(tokenize(req.Query), document.ID, document.Title, document.Summary, document.Body); queryScore > 0 {
		score += queryScore
		reasons = append(reasons, "directly matched the query text")
	}

	if len(reasons) == 0 {
		reasons = append(reasons, document.Summary)
	}

	return DocumentMatch{
		Document: document,
		Score:    score,
		Reasons:  dedupeStrings(reasons),
		Sources:  store.SourcesForIDs(document.SourceIDs),
	}, true
}

func inferContextBottleneck(req ContextRequest, gpu *GPUProfile) (string, string) {
	if req.Bottleneck != "" {
		return req.Bottleneck, ""
	}

	if req.Workload == "decode" {
		if containsAnyValue(req.Operators, "attention", "kv-cache", "paged-attention") {
			return "memory", "Fusion inferred a memory-bound decode path because the request centers on attention and KV movement."
		}
		return "latency", "Fusion inferred a latency-sensitive decode path because no heavier operator set was supplied."
	}

	if req.Workload == "prefill" {
		if containsAnyValue(req.Operators, "matmul", "gemm", "moe", "attention") {
			return "compute", "Fusion inferred a compute-bound prefill path because the request centers on tensor-core-heavy operators."
		}
	}

	if gpu != nil && canonical(gpu.Market) == "consumer" && containsAnyValue(req.Goals, "memory", "cost") {
		return "memory", "Fusion inferred a memory-bound consumer-GPU path because the request emphasizes memory or cost pressure."
	}

	return "", ""
}

func normalizeContextRequest(req ContextRequest) ContextRequest {
	req.Query = strings.TrimSpace(req.Query)
	req.GPU = strings.TrimSpace(req.GPU)
	req.Model = strings.TrimSpace(req.Model)
	req.Task = normalizeTaskContext(req.Task)
	req.Workload = firstNormalized(req.Workload)
	req.Precision = firstNormalized(req.Precision)
	req.Bottleneck = firstNormalized(req.Bottleneck)
	req.Runtime = firstNormalized(req.Runtime)
	req.Operators = normalizeValues(req.Operators)
	req.Goals = normalizeValues(req.Goals)
	if req.Workload == "" && (req.Task == "" || req.Task == "text-generation") {
		req.Workload = "decode"
	}
	return req
}

func normalizeTaskContext(value string) string {
	value = firstNormalized(value)
	switch value {
	case "", "auto":
		return ""
	case "text", "textgen", "llm", "chat", "completion":
		return "text-generation"
	case "image", "img", "imagegen":
		return "image-generation"
	case "image-edit", "edit":
		return "image-editing"
	case "video", "videogen":
		return "video-generation"
	case "audio", "tts", "speech":
		return "audio-generation"
	default:
		return value
	}
}

func taskQueryScore(task string, fields ...string) int {
	task = strings.TrimSpace(task)
	if task == "" {
		return 0
	}
	return scoreText(tokenize(task), fields...)
}

func matchesOptionalBucket(target string, values []string, allowEmpty bool) bool {
	if target == "" {
		return true
	}
	if len(values) == 0 {
		return allowEmpty
	}
	return matchesContext(target, values)
}

func matchesContext(target string, values []string) bool {
	target = canonical(target)
	if target == "" {
		return false
	}
	for _, value := range values {
		value = canonical(value)
		if value == "" {
			continue
		}
		if value == "all" || value == "*" || value == target {
			return true
		}
	}
	return false
}

func containsWildcardValues(values []string) bool {
	return matchesContext("all", values)
}

func contextOverlap(left, right []string) []string {
	left = normalizeValues(left)
	right = normalizeValues(right)

	index := map[string]struct{}{}
	for _, value := range left {
		index[value] = struct{}{}
	}

	out := []string{}
	for _, value := range right {
		if value == "all" || value == "*" {
			continue
		}
		if _, ok := index[value]; ok {
			out = append(out, value)
		}
	}
	return dedupeStrings(out)
}

func containsAnyValue(values []string, candidates ...string) bool {
	index := map[string]struct{}{}
	for _, value := range normalizeValues(values) {
		index[value] = struct{}{}
	}
	for _, candidate := range normalizeValues(candidates) {
		if _, ok := index[candidate]; ok {
			return true
		}
	}
	return false
}

func normalizeValues(values []string) []string {
	out := []string{}
	seen := map[string]struct{}{}
	for _, value := range values {
		value = firstNormalized(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func firstNormalized(value string) string {
	return strings.TrimSpace(strings.ToLower(value))
}

func dedupeStrings(values []string) []string {
	out := []string{}
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
