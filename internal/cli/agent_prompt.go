package cli

import (
	"fmt"
	"strings"

	"github.com/ModelsLab/fusion/internal/kb"
	"github.com/ModelsLab/fusion/internal/optimize"
)

func buildOptimizeRunPrompt(session *optimize.Session) string {
	if session == nil {
		return "Inspect the current project, retrieve the relevant knowledge base context, and execute a full optimization loop using Fusion tools."
	}

	lines := []string{
		"You are running a full optimization job inside Fusion.",
		"",
		"Optimization session",
		fmt.Sprintf("- id: %s", session.ID),
		fmt.Sprintf("- project_root: %s", session.ProjectRoot),
		fmt.Sprintf("- workspace_root: %s", session.WorkspaceRoot),
		fmt.Sprintf("- target: %s", valueOrFallback(session.Target, "local")),
		fmt.Sprintf("- runtime: %s", valueOrFallback(session.Runtime, "unspecified")),
		fmt.Sprintf("- user_request: %s", valueOrFallback(session.Query, defaultOptimizationQuery(session))),
		fmt.Sprintf("- model: %s", valueOrFallback(session.Request.Model, "unspecified")),
		fmt.Sprintf("- task: %s", valueOrFallback(session.Request.Task, "unspecified")),
		fmt.Sprintf("- gpu: %s", valueOrFallback(session.Request.GPU, "unspecified")),
		fmt.Sprintf("- workload: %s", valueOrFallback(session.Request.Workload, "decode")),
		fmt.Sprintf("- precision: %s", valueOrFallback(session.Request.Precision, "bf16")),
		fmt.Sprintf("- operators: %s", joinOrFallback(session.Request.Operators, "general transformer path")),
		fmt.Sprintf("- goals: %s", joinOrFallback(session.Request.Goals, "latency, throughput, memory")),
		"",
		"Required workflow",
		"1. Inspect the local project first with list_files, search_files, and read_file before deciding anything.",
		"2. Identify the runtime, inference entrypoints, benchmark scripts, tests, custom kernels, and likely hot operators from the checked-out code.",
		"3. Use detect_environment if machine capabilities matter, then retrieve or confirm the right context with build_context_packet and search_knowledge_base.",
		"4. Detect the runtime shape generically from the checked-out project. Use detect_runtime_environment when the codebase could be transformers, diffusers, vllm, sglang, or another Python runtime.",
		"5. Build an applicability matrix before running experiments. Mark each candidate family as applicable, blocked, or unsupported for this GPU, runtime, model, and task. Skip unsupported branches explicitly with a reason instead of silently ignoring them.",
		"6. Register every path you evaluate as a candidate with register_optimization_candidate. This includes baseline and runtime-only candidates such as baseline, no-attn, flash-attn, torch-compile, cuda-graphs, awq, fp8, nvfp4, kv-quant, packaged turbo or distilled checkpoints, synthesized FP8 conversion candidates, or any custom backend you choose.",
		"7. When the codebase already contains a working implementation, create a conservative verified seed before aggressive rewrites. Prefer a correctness-preserving fast path first, then optimize around bounded hot regions instead of rewriting the whole project immediately.",
		"8. Exhaust the low-hanging search ladder before custom kernels: baseline -> packaged model-family, checkpoint, or runtime flavor variants -> runtime flags or attention implementation -> dtype or quant or checkpoint variants, including synthesized FP8 conversion when no packaged FP8 artifact exists -> torch.compile or CUDA graphs if supported -> Triton or CuTe or CUDA kernels -> deeper runtime patching.",
		"9. Choose the backend and workflow yourself. Do not assume Triton, CuTe, or CUDA helpers exist. Use generic file tools plus run_command to create, edit, build, verify, and benchmark code.",
		"10. When a command belongs to a candidate stage, call run_command with session, candidate, and stage so Fusion persists the artifact. Use run_benchmark and run_profile with session and candidate for benchmark/profile stages.",
		"11. For modality-specific workloads, create a generic harness manifest with create_harness_manifest and evaluate repeated benchmark samples plus quality metrics with assess_harness. Do not force image, video, or audio workloads into tokens/sec.",
		"12. After profile collection, use analyze_profile so Fusion converts raw Nsight output into a BottleneckReport and Prescription before you decide on deeper kernel changes.",
		"13. Use infer_hotspots to map kernels back to generic model stages like attention, transformer, unet, vae, scheduler, or upscaler when the workload is not a simple text decode loop.",
		"14. Use show_outer_loop_status and record_loop_decision to make the outer-loop state explicit. Do not launch deeper custom kernel search until packaged model, runtime, quantization, compile, and attention-backend branches are exhausted or explicitly blocked.",
		"15. Run inner-loop search in phases: explore early with multiple distinct search lanes or architectural families, then exploit later around the strongest survivors. Avoid spending the whole budget on one lineage too early.",
		"16. During kernel search, persist round artifacts with save_round_artifact or record_reflexion under candidates/<id>/rounds/<n> so prompt, diagnosis, prescription, verify, bench, and reflexion data survive across turns.",
		"17. Treat failed compile, verify, or runtime candidates as useful negative examples. Record what failed and why so later rounds can avoid near-duplicate mistakes instead of rediscovering them.",
		"18. Use assess_benchmark_runs before ranking performance-sensitive candidates, and use rank_search_candidates with metadata like backend, search_lane, signature, and hypothesis so Fusion can preserve diverse survivors instead of collapsing to near-duplicates.",
		"19. Write session memory with write_session_memory after important wins, failures, blockers, or environment changes so later turns can resume from markdown evidence instead of rediscovering context.",
		"20. If compile, correctness, inference, or performance problems appear, inspect the outputs, patch the code or scripts, and retry. Do not stop at the first fixable error or the first small performance win.",
		"21. Verify correctness before claiming success. Prefer explicit tolerances, reproducible seeds, benchmark evidence, and quality checks matched to the task.",
		"22. Keep the optimization session state accurate by recording stages and using the candidate workspace instead of ad hoc temp paths.",
		"23. For FP8 or other converted quantization paths, save the calibration recipe, runtime flags, and any fallback higher-precision modules. Compare normalized steady-state metrics, not just raw wall time. Choose metrics that fit the task: tokens/sec for text, images/sec for image generation, frames/sec or clips/hour for video, and rtf or x_real_time for audio.",
		"24. Maintain a current best candidate. If a new candidate regresses or breaks correctness, fall back to the current best and continue the search.",
		"25. End only after each applicable candidate family has been tested, rejected with evidence, or blocked by the environment. Then report the best candidate, what changed, what passed, what failed, and the next most valuable experiment if more time remains.",
		"",
		"Stage guidance",
		"- Common stage names: inspect, baseline, build, verify, benchmark, profile, patch, model-benchmark, final-report.",
		"- If you need a baseline, register a baseline candidate first and attach its benchmark/profile stages there.",
	}

	if len(session.Context.Strategies) > 0 {
		lines = append(lines, "", "Top strategies")
		for _, match := range session.Context.Strategies {
			lines = append(lines, fmt.Sprintf("- %s: %s", match.Strategy.Title, match.Strategy.Summary))
		}
	}
	if len(session.Context.Skills) > 0 {
		lines = append(lines, "", "Top skills")
		for _, match := range session.Context.Skills {
			lines = append(lines, fmt.Sprintf("- %s: %s", match.Skill.Title, match.Skill.Summary))
		}
	}
	if len(session.Context.Examples) > 0 {
		lines = append(lines, "", "Relevant examples")
		for _, match := range session.Context.Examples {
			lines = append(lines, fmt.Sprintf("- %s: %s", match.Example.Title, match.Example.Summary))
		}
	}
	if len(session.Context.Documents) > 0 {
		lines = append(lines, "", "Relevant documents")
		for _, match := range session.Context.Documents {
			lines = append(lines, fmt.Sprintf("- %s: %s", match.Document.Title, match.Document.Summary))
		}
	}
	if len(session.Context.Notes) > 0 {
		lines = append(lines, "", "Context notes")
		for _, note := range session.Context.Notes {
			lines = append(lines, "- "+note)
		}
	}
	if sourceTitles := formatSourceTitles(session.Context.Sources); sourceTitles != "" {
		lines = append(lines, "", "Supporting sources", "- "+sourceTitles)
	}
	if len(session.Candidates) > 0 {
		lines = append(lines, "", "Existing candidates")
		for _, candidate := range session.Candidates {
			lines = append(lines, fmt.Sprintf("- %s [%s] at %s", candidate.Name, candidate.Backend, candidate.Workspace))
		}
	}

	return strings.Join(lines, "\n")
}

func formatSourceTitles(sources []kb.Source) string {
	out := make([]string, 0, len(sources))
	for _, source := range sources {
		title := strings.TrimSpace(source.Title)
		if title != "" {
			out = append(out, title)
		}
	}
	return strings.Join(out, ", ")
}
