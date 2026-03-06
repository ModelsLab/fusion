package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelsLabClientCompleteUsesModelIDAndParsesToolCalls(t *testing.T) {
	var captured map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("unexpected authorization header %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "kernel updated",
						"tool_calls": []map[string]any{
							{
								"id":   "call-1",
								"type": "function",
								"function": map[string]any{
									"name":      "read_file",
									"arguments": "{\"path\":\"kernel.py\"}",
								},
							},
						},
					},
				},
			},
		})
	}))
	defer server.Close()

	client := &modelsLabClient{
		baseURL: server.URL,
		token:   "test-key",
	}

	resp, err := client.Complete(context.Background(), CompletionRequest{
		Model:        "openai-gpt-5.4-pro",
		SystemPrompt: "system prompt",
		Messages: []Message{
			{Role: "user", Content: "optimize this"},
		},
		Tools: []ToolDefinition{
			{
				Name:        "read_file",
				Description: "Read a file",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		MaxTokens: 512,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if captured["model_id"] != "openai-gpt-5.4-pro" {
		t.Fatalf("expected model_id, got %#v", captured["model_id"])
	}
	if captured["stream"] != true {
		t.Fatalf("expected stream=true, got %#v", captured["stream"])
	}
	if _, hasLegacyModelField := captured["model"]; hasLegacyModelField {
		t.Fatalf("did not expect legacy model field in request payload")
	}
	if resp.Text != "kernel updated" {
		t.Fatalf("expected response text %q, got %q", "kernel updated", resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "read_file" {
		t.Fatalf("expected one read_file tool call, got %#v", resp.ToolCalls)
	}
}

func TestModelsLabClientCompleteSurfacesModelsLabErrorMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":  "error",
			"message": "Provider returned error",
		})
	}))
	defer server.Close()

	client := &modelsLabClient{
		baseURL: server.URL,
		token:   "test-key",
	}

	_, err := client.Complete(context.Background(), CompletionRequest{
		Model: "openai-gpt-5.4-pro",
		Messages: []Message{
			{Role: "user", Content: "optimize this"},
		},
	})
	if err == nil {
		t.Fatal("expected Complete() to surface the ModelsLab error message")
	}
	if !strings.Contains(err.Error(), "Provider returned error") {
		t.Fatalf("expected ModelsLab message in error, got %v", err)
	}
}

func TestModelsLabClientCompleteParsesStreamedTextAndToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		chunks := []string{
			": OPENROUTER PROCESSING\n\n",
			`data: {"choices":[{"delta":{"content":"Kernel "}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"updated."}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\""}}]}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"path\":\"README.md\"}"}}]}}]}` + "\n\n",
			"data: [DONE]\n\n",
		}

		for _, chunk := range chunks {
			if _, err := fmt.Fprint(w, chunk); err != nil {
				t.Fatalf("write chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := &modelsLabClient{
		baseURL: server.URL,
		token:   "test-key",
	}

	resp, err := client.Complete(context.Background(), CompletionRequest{
		Model: "openai-gpt-5.4-pro",
		Messages: []Message{
			{Role: "user", Content: "optimize this"},
		},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if resp.Text != "Kernel updated." {
		t.Fatalf("expected streamed text %q, got %q", "Kernel updated.", resp.Text)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %#v", resp.ToolCalls)
	}
	if resp.ToolCalls[0].Name != "read_file" {
		t.Fatalf("expected tool name %q, got %#v", "read_file", resp.ToolCalls[0])
	}
	if resp.ToolCalls[0].Arguments != "{\"path\":\"README.md\"}" {
		t.Fatalf("expected streamed arguments, got %q", resp.ToolCalls[0].Arguments)
	}
}
