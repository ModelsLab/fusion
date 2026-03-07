package optimize

import "testing"

func TestInferHotspotAttribution(t *testing.T) {
	report := InferHotspotAttribution("video-generation", "python", "sampling", []string{
		"flash_attention_fwd",
		"vae_decoder_kernel",
		"upsample_block_kernel",
	})
	if len(report.Matches) != 3 {
		t.Fatalf("expected 3 hotspot matches, got %+v", report)
	}
	if report.Matches[0].Component != "attention" {
		t.Fatalf("expected attention component, got %+v", report.Matches[0])
	}
	if report.Matches[1].Component != "vae" {
		t.Fatalf("expected vae component, got %+v", report.Matches[1])
	}
	if report.Matches[2].Component != "upscaler" {
		t.Fatalf("expected upscaler component, got %+v", report.Matches[2])
	}
}
