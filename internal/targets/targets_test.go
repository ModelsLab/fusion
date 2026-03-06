package targets

import (
	"testing"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/kb"
)

func TestValidateSimTargetWarnsButDoesNotFail(t *testing.T) {
	store, err := kb.LoadDefault()
	if err != nil {
		t.Fatalf("LoadDefault() error = %v", err)
	}

	validation := Validate(config.TargetConfig{
		Name:     "sim-h100",
		Mode:     "sim",
		GPU:      "h100",
		ProxyGPU: "rtx4090",
	}, store)

	if len(validation.Errors) != 0 {
		t.Fatalf("expected no validation errors, got %v", validation.Errors)
	}
	if len(validation.Warnings) == 0 {
		t.Fatal("expected sim validation warnings")
	}
}

func TestValidateSSHTargetRequiresHost(t *testing.T) {
	validation := Validate(config.TargetConfig{
		Name: "remote",
		Mode: "ssh",
	}, nil)

	if len(validation.Errors) == 0 {
		t.Fatal("expected ssh target validation to fail without host")
	}
}
