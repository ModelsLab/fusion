package kb

type Source struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	URL          string   `json:"url"`
	Type         string   `json:"type"`
	Category     string   `json:"category"`
	Reliability  string   `json:"reliability"`
	ReviewStatus string   `json:"review_status"`
	Summary      string   `json:"summary"`
	Tags         []string `json:"tags"`
}

type GPUProfile struct {
	ID                    string   `json:"id"`
	Name                  string   `json:"name"`
	Aliases               []string `json:"aliases"`
	Family                string   `json:"family"`
	Market                string   `json:"market"`
	ComputeCapability     string   `json:"compute_capability"`
	MemoryGB              int      `json:"memory_gb"`
	MemoryBandwidthGBps   int      `json:"memory_bandwidth_gbps"`
	PreferredPrecisions   []string `json:"preferred_precisions"`
	ExperimentalPrecisons []string `json:"experimental_precisions"`
	Strengths             []string `json:"strengths"`
	Constraints           []string `json:"constraints"`
	SourceIDs             []string `json:"source_ids"`
}

type Strategy struct {
	ID            string   `json:"id"`
	Title         string   `json:"title"`
	Category      string   `json:"category"`
	Summary       string   `json:"summary"`
	SupportLevel  string   `json:"support_level"`
	Workloads     []string `json:"workloads"`
	Operators     []string `json:"operators"`
	GPUFamilies   []string `json:"gpu_families"`
	GPUIDs        []string `json:"gpu_ids"`
	Precision     []string `json:"precision"`
	Bottlenecks   []string `json:"bottlenecks"`
	Goals         []string `json:"goals"`
	Priority      int      `json:"priority"`
	Preconditions []string `json:"preconditions"`
	Actions       []string `json:"actions"`
	Metrics       []string `json:"metrics"`
	Tradeoffs     []string `json:"tradeoffs"`
	SourceIDs     []string `json:"source_ids"`
}

type SkillTrigger struct {
	GPUFamilies []string `json:"gpu_families"`
	GPUIDs      []string `json:"gpu_ids"`
	Workloads   []string `json:"workloads"`
	Operators   []string `json:"operators"`
	Precision   []string `json:"precision"`
	Bottlenecks []string `json:"bottlenecks"`
	Runtimes    []string `json:"runtimes"`
	Goals       []string `json:"goals"`
}

type Skill struct {
	ID                 string       `json:"id"`
	Title              string       `json:"title"`
	Category           string       `json:"category"`
	Summary            string       `json:"summary"`
	SupportLevel       string       `json:"support_level"`
	Triggers           SkillTrigger `json:"triggers"`
	PreferredBackends  []string     `json:"preferred_backends"`
	RequiredTools      []string     `json:"required_tools"`
	Steps              []string     `json:"steps"`
	Verification       []string     `json:"verification"`
	BenchmarkRubric    []string     `json:"benchmark_rubric"`
	FailureRecovery    []string     `json:"failure_recovery"`
	ArtifactsToSave    []string     `json:"artifacts_to_save"`
	RuntimeAdapters    []string     `json:"runtime_adapters"`
	ReferenceSourceIDs []string     `json:"reference_source_ids"`
}

type Example struct {
	ID             string   `json:"id"`
	Title          string   `json:"title"`
	Category       string   `json:"category"`
	Backend        string   `json:"backend"`
	Summary        string   `json:"summary"`
	SupportLevel   string   `json:"support_level"`
	GPUFamilies    []string `json:"gpu_families"`
	GPUIDs         []string `json:"gpu_ids"`
	Workloads      []string `json:"workloads"`
	Operators      []string `json:"operators"`
	Precision      []string `json:"precision"`
	Runtimes       []string `json:"runtimes"`
	UseCases       []string `json:"use_cases"`
	Notes          []string `json:"notes"`
	ReferencePaths []string `json:"reference_paths"`
	SourceIDs      []string `json:"source_ids"`
}

type Document struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	Category     string   `json:"category"`
	Summary      string   `json:"summary"`
	SupportLevel string   `json:"support_level"`
	Reliability  string   `json:"reliability"`
	ReviewStatus string   `json:"review_status"`
	URL          string   `json:"url,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	GPUFamilies  []string `json:"gpu_families,omitempty"`
	GPUIDs       []string `json:"gpu_ids,omitempty"`
	Workloads    []string `json:"workloads,omitempty"`
	Operators    []string `json:"operators,omitempty"`
	Precision    []string `json:"precision,omitempty"`
	Runtimes     []string `json:"runtimes,omitempty"`
	Backends     []string `json:"backends,omitempty"`
	SourceIDs    []string `json:"source_ids,omitempty"`
	Path         string   `json:"path"`
	Body         string   `json:"body"`
}

type StrategyMatch struct {
	Strategy Strategy `json:"strategy"`
	Score    int      `json:"score"`
	Reasons  []string `json:"reasons"`
	Sources  []Source `json:"sources"`
}

type SkillMatch struct {
	Skill   Skill    `json:"skill"`
	Score   int      `json:"score"`
	Reasons []string `json:"reasons"`
	Sources []Source `json:"sources"`
}

type ExampleMatch struct {
	Example Example  `json:"example"`
	Score   int      `json:"score"`
	Reasons []string `json:"reasons"`
	Sources []Source `json:"sources"`
}

type DocumentMatch struct {
	Document Document `json:"document"`
	Score    int      `json:"score"`
	Reasons  []string `json:"reasons"`
	Sources  []Source `json:"sources"`
}

type ContextRequest struct {
	Query               string   `json:"query"`
	GPU                 string   `json:"gpu"`
	Model               string   `json:"model"`
	Task                string   `json:"task"`
	Workload            string   `json:"workload"`
	Operators           []string `json:"operators"`
	Precision           string   `json:"precision"`
	Bottleneck          string   `json:"bottleneck"`
	Runtime             string   `json:"runtime"`
	Goals               []string `json:"goals"`
	IncludeExperimental bool     `json:"include_experimental"`
	Limit               int      `json:"limit"`
}

type ContextPacket struct {
	Request    ContextRequest  `json:"request"`
	GPU        *GPUProfile     `json:"gpu,omitempty"`
	Strategies []StrategyMatch `json:"strategies"`
	Skills     []SkillMatch    `json:"skills"`
	Examples   []ExampleMatch  `json:"examples"`
	Documents  []DocumentMatch `json:"documents"`
	Sources    []Source        `json:"sources"`
	Notes      []string        `json:"notes"`
}

type IndexRecord struct {
	Kind         string `json:"kind"`
	ID           string `json:"id"`
	Title        string `json:"title"`
	Summary      string `json:"summary"`
	Body         string `json:"body"`
	Category     string `json:"category"`
	SupportLevel string `json:"support_level"`
	Reliability  string `json:"reliability"`
	ReviewStatus string `json:"review_status"`
	JSON         string `json:"json"`
}

type SearchHit struct {
	Kind    string
	ID      string
	Title   string
	Summary string
	Score   int
}
