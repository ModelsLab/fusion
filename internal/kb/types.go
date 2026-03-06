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

type SearchHit struct {
	Kind    string
	ID      string
	Title   string
	Summary string
	Score   int
}
