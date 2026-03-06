package system

import (
	"strings"
	"testing"
)

func TestNormalizeDoctorBackend(t *testing.T) {
	cases := map[string]string{
		"":         "all",
		"all":      "all",
		"cuda":     "cuda",
		"triton":   "triton",
		"cute":     "cute",
		"cute_dsl": "cute",
		"nsight":   "profile",
		"weird":    "all",
	}

	for input, expected := range cases {
		if got := normalizeDoctorBackend(input); got != expected {
			t.Fatalf("normalizeDoctorBackend(%q) = %q, want %q", input, got, expected)
		}
	}
}

func TestRecommendedFixScriptIncludesBackendPackages(t *testing.T) {
	report := DoctorReport{
		Backend: "triton",
		Checks: []DoctorCheck{
			{ID: "python3", OK: false},
			{ID: "module:torch", OK: false},
			{ID: "module:triton", OK: false},
			{ID: "uv", OK: false},
			{ID: "nvidia-smi", OK: false},
		},
	}

	script := RecommendedFixScript(report)
	for _, needle := range []string{
		"sudo apt-get update",
		"python3 -m pip install --upgrade pip",
		"torch",
		"triton",
		"uv",
		"nvidia-smi",
	} {
		if !strings.Contains(script, needle) {
			t.Fatalf("expected fix script to contain %q, got:\n%s", needle, script)
		}
	}
}

func TestDoctorRequirementsDeduplicatesCommonEntries(t *testing.T) {
	requirements := doctorRequirements("all")
	seen := map[string]struct{}{}
	for _, requirement := range requirements {
		if _, ok := seen[requirement.id]; ok {
			t.Fatalf("duplicate requirement id %q", requirement.id)
		}
		seen[requirement.id] = struct{}{}
	}
}
