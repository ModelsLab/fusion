package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/artifacts"
	"github.com/ModelsLab/fusion/internal/optimize"
)

func showOuterLoopStatusTool() Tool {
	type input struct {
		Session string `json:"session"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "show_outer_loop_status",
			Description: "Show whether the outer optimization loop is exhausted and the inner kernel loop is ready to start.",
			InputSchema: objectSchema(
				map[string]any{
					"session": stringSchema("optimization session id"),
				},
				[]string{"session"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			session, _, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}
			return marshalPretty(optimize.EvaluateOuterLoopStatus(session))
		},
	}
}

func recordLoopDecisionTool() Tool {
	type input struct {
		Session    string `json:"session"`
		Phase      string `json:"phase"`
		Family     string `json:"family"`
		Status     string `json:"status"`
		Candidate  string `json:"candidate"`
		Reason     string `json:"reason"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "record_loop_decision",
			Description: "Record an explicit outer-loop or inner-loop decision like tested, blocked, skipped, regressed, or winner.",
			InputSchema: objectSchema(
				map[string]any{
					"session":   stringSchema("optimization session id"),
					"phase":     stringSchema("loop phase, for example outer or inner"),
					"family":    stringSchema("decision family like baseline, model-family, runtime, quantization, compile, or attention-backend"),
					"status":    stringSchema("decision status like tested, blocked, skipped, regressed, or winner"),
					"candidate": stringSchema("optional candidate id"),
					"reason":    stringSchema("short explanation for the decision"),
				},
				[]string{"session", "family", "status"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			session, store, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}
			session.RecordLoopDecision(valueOrDefault(req.Phase, "outer"), req.Family, req.Status, req.Candidate, req.Reason)
			if strings.EqualFold(strings.TrimSpace(req.Status), "winner") && strings.TrimSpace(req.Candidate) != "" {
				session.SetCurrentBestCandidate(req.Candidate)
			}
			if _, err := store.Save(session); err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"session":   session.ID,
				"phase":     valueOrDefault(req.Phase, "outer"),
				"family":    req.Family,
				"status":    req.Status,
				"candidate": strings.TrimSpace(req.Candidate),
				"recorded":  true,
			})
		},
	}
}

func analyzeProfileTool(toolCtx ToolContext) Tool {
	type input struct {
		Artifact string `json:"artifact"`
		Tool     string `json:"tool"`
		Stdout   string `json:"stdout"`
		Stderr   string `json:"stderr"`
		Session  string `json:"session"`
		Candidate string `json:"candidate"`
		Round    int    `json:"round"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "analyze_profile",
			Description: "Parse raw or saved Nsight output into normalized metrics, a bottleneck report, and a prescription. Optionally persist the round artifacts.",
			InputSchema: objectSchema(
				map[string]any{
					"artifact":  stringSchema("optional path to a saved profile artifact JSON"),
					"tool":      stringSchema("profiler tool label like ncu or nsys"),
					"stdout":    stringSchema("raw profiler stdout when no artifact path is provided"),
					"stderr":    stringSchema("raw profiler stderr when no artifact path is provided"),
					"session":   stringSchema("optional optimization session id"),
					"candidate": stringSchema("optional candidate id when saving round artifacts"),
					"round":     intSchema("optional inner-loop round number when saving artifacts"),
				},
				nil,
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			stdout := req.Stdout
			stderr := req.Stderr
			tool := req.Tool
			if strings.TrimSpace(req.Artifact) != "" {
				path, err := resolvePath(toolCtx.CWD, req.Artifact)
				if err != nil {
					return "", err
				}
				store, err := artifacts.NewStore()
				if err != nil {
					return "", err
				}
				artifact, err := store.LoadProfile(path)
				if err != nil {
					return "", err
				}
				stdout = artifact.Stdout
				stderr = artifact.Stderr
				if strings.TrimSpace(tool) == "" {
					tool = artifact.Tool
				}
				req.Artifact = path
			}

			profile := optimize.ParseNsightProfile(tool, stdout, stderr)
			report := optimize.AnalyzeRoofline(profile)
			prescription := optimize.PrescribeFromReport(report, optimize.Request{}, optimize.Candidate{
				ID:      strings.TrimSpace(req.Candidate),
				Backend: profile.Tool,
			})

			payload := map[string]any{
				"artifact":      strings.TrimSpace(req.Artifact),
				"profile":       profile,
				"diagnosis":     report,
				"prescription":  prescription,
			}
			if strings.TrimSpace(req.Session) != "" && strings.TrimSpace(req.Candidate) != "" && req.Round > 0 {
				session, _, err := loadOptimizationSession(req.Session)
				if err != nil {
					return "", err
				}
				profilePath, err := optimize.SaveRoundArtifact(session, req.Candidate, req.Round, "profile", profile)
				if err != nil {
					return "", err
				}
				diagnosisPath, err := optimize.SaveRoundArtifact(session, req.Candidate, req.Round, "diagnosis", report)
				if err != nil {
					return "", err
				}
				prescriptionPath, err := optimize.SaveRoundArtifact(session, req.Candidate, req.Round, "prescription", prescription)
				if err != nil {
					return "", err
				}
				payload["round_artifacts"] = map[string]string{
					"profile":      profilePath,
					"diagnosis":    diagnosisPath,
					"prescription": prescriptionPath,
				}
			}
			return marshalPretty(payload)
		},
	}
}

func assessBenchmarkRunsTool() Tool {
	type sample struct {
		Name       string             `json:"name"`
		Metrics    map[string]float64 `json:"metrics"`
		DurationMS int64              `json:"duration_ms"`
		ExitCode   int                `json:"exit_code"`
		Warmup     bool               `json:"warmup"`
	}
	type input struct {
		WarmupRuns        int      `json:"warmup_runs"`
		MeasuredRuns      int      `json:"measured_runs"`
		VarianceThreshold float64  `json:"variance_threshold"`
		NormalizedMetrics []string `json:"normalized_metrics"`
		LockName          string   `json:"lock_name"`
		Samples           []sample `json:"samples"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "assess_benchmark_runs",
			Description: "Apply Fusion's benchmark protocol to repeated runs and report whether the benchmark is stable enough to rank.",
			InputSchema: objectSchema(
				map[string]any{
					"warmup_runs":        intSchema("number of warmup runs"),
					"measured_runs":      intSchema("number of measured runs"),
					"variance_threshold": numberSchema("maximum allowed relative standard deviation"),
					"normalized_metrics": map[string]any{"type": "array", "items": stringSchema("normalized metric name")},
					"lock_name":          stringSchema("optional benchmark lock label"),
					"samples": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name":        stringSchema("sample name"),
								"metrics":     map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
								"duration_ms": intSchema("sample duration"),
								"exit_code":   intSchema("exit code"),
								"warmup":      boolSchema("whether the sample was a warmup run"),
							},
						},
					},
				},
				[]string{"samples"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			protocol := optimize.BenchmarkProtocol{
				WarmupRuns:        req.WarmupRuns,
				MeasuredRuns:      req.MeasuredRuns,
				VarianceThreshold: req.VarianceThreshold,
				NormalizedMetrics: append([]string{}, req.NormalizedMetrics...),
				LockName:          req.LockName,
			}
			samples := make([]optimize.BenchmarkRunSample, 0, len(req.Samples))
			for _, sample := range req.Samples {
				samples = append(samples, optimize.BenchmarkRunSample{
					Name:       sample.Name,
					Metrics:    sample.Metrics,
					DurationMS: sample.DurationMS,
					ExitCode:   sample.ExitCode,
					Warmup:     sample.Warmup,
				})
			}
			return marshalPretty(protocol.Evaluate(samples))
		},
	}
}

func rankSearchCandidatesTool() Tool {
	type metricSummary struct {
		Mean           float64 `json:"mean"`
		StdDev         float64 `json:"stddev"`
		RelativeStdDev float64 `json:"relative_stddev"`
		Samples        int     `json:"samples"`
	}
	type assessment struct {
		Stable        bool                     `json:"stable"`
		PrimaryMetric string                   `json:"primary_metric"`
		MetricStats   map[string]metricSummary `json:"metric_stats"`
	}
	type state struct {
		CandidateID string             `json:"candidate_id"`
		Round       int                `json:"round"`
		Verified    bool               `json:"verified"`
		BuildPassed bool               `json:"build_passed"`
		Efficiency  float64            `json:"efficiency"`
		Metrics     map[string]float64 `json:"metrics"`
		Assessment  assessment         `json:"assessment"`
	}
	type input struct {
		Session     string `json:"session"`
		PromoteBest bool   `json:"promote_best"`
		Mode        string `json:"mode"`
		BeamWidth   int    `json:"beam_width"`
		States      []state `json:"states"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "rank_search_candidates",
			Description: "Rank inner-loop search branches with Fusion's search manager and optionally promote the top candidate as the current best.",
			InputSchema: objectSchema(
				map[string]any{
					"session":      stringSchema("optional optimization session id"),
					"promote_best": boolSchema("update the session winner with the top-ranked candidate"),
					"mode":         stringSchema("search mode like greedy, beam, or bandit"),
					"beam_width":   intSchema("beam width when using beam or bandit search"),
					"states": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"candidate_id": stringSchema("candidate id"),
								"round":        intSchema("search round"),
								"verified":     boolSchema("whether verify passed"),
								"build_passed": boolSchema("whether build passed"),
								"efficiency":   numberSchema("roofline efficiency between 0 and 1"),
								"metrics":      map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "number"}},
								"assessment": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"stable":         boolSchema("benchmark stability verdict"),
										"primary_metric": stringSchema("primary metric name"),
										"metric_stats": map[string]any{
											"type": "object",
											"additionalProperties": map[string]any{
												"type": "object",
												"properties": map[string]any{
													"mean":             numberSchema("mean value"),
													"stddev":           numberSchema("stddev"),
													"relative_stddev":  numberSchema("relative stddev"),
													"samples":          intSchema("sample count"),
												},
											},
										},
									},
								},
							},
						},
					},
				},
				[]string{"states"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			manager := optimize.SearchManager{
				Config: optimize.SearchConfig{
					Mode:      optimize.SearchMode(strings.TrimSpace(strings.ToLower(req.Mode))),
					BeamWidth: req.BeamWidth,
				},
			}
			states := make([]optimize.SearchCandidateState, 0, len(req.States))
			for _, state := range req.States {
				metricStats := map[string]optimize.BenchmarkMetricSummary{}
				for key, summary := range state.Assessment.MetricStats {
					metricStats[key] = optimize.BenchmarkMetricSummary{
						Mean:           summary.Mean,
						StdDev:         summary.StdDev,
						RelativeStdDev: summary.RelativeStdDev,
						Samples:        summary.Samples,
					}
				}
				states = append(states, optimize.SearchCandidateState{
					CandidateID: state.CandidateID,
					Round:       state.Round,
					Verified:    state.Verified,
					BuildPassed: state.BuildPassed,
					Efficiency:  state.Efficiency,
					Metrics:     state.Metrics,
					Assessment: optimize.BenchmarkAssessment{
						Stable:        state.Assessment.Stable,
						PrimaryMetric: state.Assessment.PrimaryMetric,
						MetricStats:   metricStats,
					},
				})
			}
			selection := manager.Select(states)
			payload := map[string]any{
				"selection": selection,
			}
			if strings.TrimSpace(req.Session) != "" && req.PromoteBest && len(selection.Ranked) > 0 {
				session, store, err := loadOptimizationSession(req.Session)
				if err != nil {
					return "", err
				}
				session.SetCurrentBestCandidate(selection.Ranked[0].CandidateID)
				session.InnerLoop.Status = "active"
				session.InnerLoop.SearchMode = string(selection.Config.Mode)
				session.InnerLoop.BeamWidth = selection.Config.BeamWidth
				session.InnerLoop.CurrentRound = selection.Ranked[0].Round
				session.InnerLoop.BestCandidateID = selection.Ranked[0].CandidateID
				session.InnerLoop.UpdatedAt = time.Now().UTC()
				if session.InnerLoop.StartedAt.IsZero() {
					session.InnerLoop.StartedAt = session.InnerLoop.UpdatedAt
				}
				if _, err := store.Save(session); err != nil {
					return "", err
				}
				payload["promoted_best_candidate"] = selection.Ranked[0].CandidateID
			}
			return marshalPretty(payload)
		},
	}
}

func saveRoundArtifactTool(toolCtx ToolContext) Tool {
	type input struct {
		Session     string `json:"session"`
		Candidate   string `json:"candidate"`
		Round       int    `json:"round"`
		Kind        string `json:"kind"`
		Content     string `json:"content"`
		JSONContent string `json:"json_content"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "save_round_artifact",
			Description: "Persist a round artifact under candidates/<id>/rounds/<n>/<kind>.json so the inner loop has a stable artifact layout.",
			InputSchema: objectSchema(
				map[string]any{
					"session":      stringSchema("optimization session id"),
					"candidate":    stringSchema("candidate id"),
					"round":        intSchema("round number starting at 1"),
					"kind":         stringSchema("artifact kind like prompt, reply, code, verify, bench, diagnosis, prescription, or reflexion"),
					"content":      stringSchema("plain text content to wrap as JSON"),
					"json_content": stringSchema("raw JSON payload to save instead of plain text"),
				},
				[]string{"session", "candidate", "round", "kind"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			session, _, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}
			var payload any = map[string]any{"content": req.Content}
			if strings.TrimSpace(req.JSONContent) != "" {
				var decoded any
				if err := json.Unmarshal([]byte(req.JSONContent), &decoded); err != nil {
					return "", fmt.Errorf("decode json_content: %w", err)
				}
				payload = decoded
			}
			path, err := optimize.SaveRoundArtifact(session, req.Candidate, req.Round, req.Kind, payload)
			if err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"session":   session.ID,
				"candidate": req.Candidate,
				"round":     req.Round,
				"kind":      req.Kind,
				"path":      path,
			})
		},
	}
}

func recordReflexionTool(toolCtx ToolContext) Tool {
	type input struct {
		Session       string   `json:"session"`
		Candidate     string   `json:"candidate"`
		Round         int      `json:"round"`
		Hypothesis    string   `json:"hypothesis"`
		Outcome       string   `json:"outcome"`
		Effective     bool     `json:"effective"`
		Lessons       []string `json:"lessons"`
		AvoidPatterns []string `json:"avoid_patterns"`
		TryNext       []string `json:"try_next"`
	}

	return Tool{
		Definition: ToolDefinition{
			Name:        "record_reflexion",
			Description: "Save a structured round-by-round reflexion artifact for later search, ranking, and KB promotion.",
			InputSchema: objectSchema(
				map[string]any{
					"session":        stringSchema("optimization session id"),
					"candidate":      stringSchema("candidate id"),
					"round":          intSchema("round number"),
					"hypothesis":     stringSchema("what this round tried to prove"),
					"outcome":        stringSchema("what happened"),
					"effective":      boolSchema("whether the change helped"),
					"lessons":        map[string]any{"type": "array", "items": stringSchema("lesson")},
					"avoid_patterns": map[string]any{"type": "array", "items": stringSchema("pattern to avoid")},
					"try_next":       map[string]any{"type": "array", "items": stringSchema("next action")},
				},
				[]string{"session", "candidate", "round"},
			),
		},
		Execute: func(ctx context.Context, arguments string) (string, error) {
			var req input
			if err := decodeArguments(arguments, &req); err != nil {
				return "", err
			}
			session, _, err := loadOptimizationSession(req.Session)
			if err != nil {
				return "", err
			}
			reflexion := optimize.Reflexion{
				Version:       1,
				Round:         req.Round,
				Hypothesis:    strings.TrimSpace(req.Hypothesis),
				Outcome:       strings.TrimSpace(req.Outcome),
				Effective:     req.Effective,
				Lessons:       append([]string{}, req.Lessons...),
				AvoidPatterns: append([]string{}, req.AvoidPatterns...),
				TryNext:       append([]string{}, req.TryNext...),
				RecordedAt:    time.Now().UTC(),
			}
			path, err := optimize.SaveRoundArtifact(session, req.Candidate, req.Round, "reflexion", reflexion)
			if err != nil {
				return "", err
			}
			return marshalPretty(map[string]any{
				"session":   session.ID,
				"candidate": req.Candidate,
				"round":     req.Round,
				"path":      path,
				"reflexion": reflexion,
			})
		},
	}
}
