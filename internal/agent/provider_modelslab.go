package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ModelsLab/fusion/internal/modelslab"
)

type modelsLabClient struct {
	baseURL string
	token   string
	client  *http.Client
}

type modelsLabChatRequest struct {
	ModelID    string             `json:"model_id"`
	Messages   []modelsLabMessage `json:"messages"`
	Tools      []modelsLabTool    `json:"tools,omitempty"`
	ToolChoice string             `json:"tool_choice,omitempty"`
	MaxTokens  int                `json:"max_tokens,omitempty"`
	Stream     bool               `json:"stream"`
}

type modelsLabMessage struct {
	Role       string              `json:"role"`
	Content    any                 `json:"content,omitempty"`
	ToolCalls  []modelsLabToolCall `json:"tool_calls,omitempty"`
	ToolCallID string              `json:"tool_call_id,omitempty"`
}

type modelsLabTool struct {
	Type     string                  `json:"type"`
	Function modelsLabFunctionSchema `json:"function"`
}

type modelsLabFunctionSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type modelsLabToolCall struct {
	ID       string                `json:"id,omitempty"`
	Type     string                `json:"type"`
	Function modelsLabFunctionCall `json:"function"`
}

type modelsLabFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type modelsLabChatResponse struct {
	Status  string `json:"status,omitempty"`
	Message string `json:"message,omitempty"`
	Choices []struct {
		Message modelsLabChoiceMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type modelsLabChoiceMessage struct {
	Content   any                 `json:"content"`
	ToolCalls []modelsLabToolCall `json:"tool_calls,omitempty"`
}

func (c *modelsLabClient) Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error) {
	httpClient := c.client
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	messages := make([]modelsLabMessage, 0, len(req.Messages)+1)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		messages = append(messages, modelsLabMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}
	for _, message := range req.Messages {
		switch message.Role {
		case "user":
			messages = append(messages, modelsLabMessage{
				Role:    "user",
				Content: message.Content,
			})
		case "assistant":
			toolCalls := make([]modelsLabToolCall, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				toolCalls = append(toolCalls, modelsLabToolCall{
					ID:   call.ID,
					Type: "function",
					Function: modelsLabFunctionCall{
						Name:      call.Name,
						Arguments: call.Arguments,
					},
				})
			}
			messages = append(messages, modelsLabMessage{
				Role:      "assistant",
				Content:   nullableString(message.Content),
				ToolCalls: toolCalls,
			})
		case "tool":
			messages = append(messages, modelsLabMessage{
				Role:       "tool",
				Content:    message.Content,
				ToolCallID: message.ToolCallID,
			})
		}
	}

	tools := make([]modelsLabTool, 0, len(req.Tools))
	for _, tool := range req.Tools {
		tools = append(tools, modelsLabTool{
			Type: "function",
			Function: modelsLabFunctionSchema{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	payload := modelsLabChatRequest{
		ModelID:    req.Model,
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
		MaxTokens:  req.MaxTokens,
		Stream:     true,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("encode modelslab request: %w", err)
	}

	baseURL := c.baseURL
	if strings.TrimSpace(baseURL) == "" {
		baseURL = modelslab.APIBaseURL()
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/chat/completions"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("build modelslab request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("call modelslab API: %w", err)
	}
	defer resp.Body.Close()

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	bodyReader := newMaxBytesReader(resp.Body, maxModelsLabResponseBytes)
	if strings.Contains(contentType, "text/event-stream") {
		if resp.StatusCode >= 400 {
			body, readErr := io.ReadAll(bodyReader)
			if readErr != nil {
				return CompletionResponse{}, fmt.Errorf("read modelslab stream error response: %w", readErr)
			}
			return CompletionResponse{}, decodeModelsLabResponseError(body, resp.Status)
		}
		return decodeModelsLabEventStream(bodyReader)
	}

	body, err := io.ReadAll(bodyReader)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("read modelslab response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return CompletionResponse{}, decodeModelsLabResponseError(body, resp.Status)
	}
	return decodeModelsLabResponse(body)
}

type modelsLabStreamChunk struct {
	Status  string `json:"status,omitempty"`
	Message string `json:"message,omitempty"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
	Choices []struct {
		Delta struct {
			Content   any                        `json:"content,omitempty"`
			ToolCalls []modelsLabStreamToolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

type modelsLabStreamToolCall struct {
	Index    int                   `json:"index,omitempty"`
	ID       string                `json:"id,omitempty"`
	Type     string                `json:"type,omitempty"`
	Function modelsLabFunctionCall `json:"function"`
}

type streamedToolCall struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func decodeModelsLabEventStream(r io.Reader) (CompletionResponse, error) {
	reader := bufio.NewReader(r)
	var (
		text      strings.Builder
		eventData []string
		toolCalls = map[int]*streamedToolCall{}
	)

	processEvent := func() error {
		if len(eventData) == 0 {
			return nil
		}
		payload := strings.TrimSpace(strings.Join(eventData, "\n"))
		eventData = eventData[:0]
		if payload == "" {
			return nil
		}
		if payload == "[DONE]" {
			return io.EOF
		}

		var chunk modelsLabStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode modelslab stream chunk: %w", err)
		}
		if strings.EqualFold(strings.TrimSpace(chunk.Status), "error") {
			if strings.TrimSpace(chunk.Message) != "" {
				return fmt.Errorf("modelslab API error: %s", chunk.Message)
			}
			if chunk.Error != nil && strings.TrimSpace(chunk.Error.Message) != "" {
				return fmt.Errorf("modelslab API error: %s", chunk.Error.Message)
			}
			return fmt.Errorf("modelslab API error: unknown error response")
		}

		for _, choice := range chunk.Choices {
			if deltaText := extractModelslabText(choice.Delta.Content); deltaText != "" {
				text.WriteString(deltaText)
			}
			for _, deltaCall := range choice.Delta.ToolCalls {
				call := toolCalls[deltaCall.Index]
				if call == nil {
					call = &streamedToolCall{}
					toolCalls[deltaCall.Index] = call
				}
				if deltaCall.ID != "" {
					call.ID = deltaCall.ID
				}
				if deltaCall.Function.Name != "" {
					call.Name += deltaCall.Function.Name
				}
				if deltaCall.Function.Arguments != "" {
					call.Arguments.WriteString(deltaCall.Function.Arguments)
				}
			}
		}
		return nil
	}

	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return CompletionResponse{}, fmt.Errorf("read modelslab event stream: %w", err)
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if procErr := processEvent(); procErr != nil {
				if procErr == io.EOF {
					break
				}
				return CompletionResponse{}, procErr
			}
		} else if strings.HasPrefix(line, "data:") {
			eventData = append(eventData, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}

		if err == io.EOF {
			if procErr := processEvent(); procErr != nil && procErr != io.EOF {
				return CompletionResponse{}, procErr
			}
			break
		}
	}

	return CompletionResponse{
		Text:      text.String(),
		ToolCalls: finalizeStreamedToolCalls(toolCalls),
	}, nil
}

func finalizeStreamedToolCalls(calls map[int]*streamedToolCall) []ToolCall {
	if len(calls) == 0 {
		return nil
	}
	maxIndex := -1
	for index := range calls {
		if index > maxIndex {
			maxIndex = index
		}
	}
	ordered := make([]ToolCall, 0, len(calls))
	for index := 0; index <= maxIndex; index++ {
		call := calls[index]
		if call == nil {
			continue
		}
		ordered = append(ordered, ToolCall{
			ID:        call.ID,
			Name:      call.Name,
			Arguments: call.Arguments.String(),
		})
	}
	return ordered
}

func decodeModelsLabResponse(body []byte) (CompletionResponse, error) {
	var decoded modelsLabChatResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return CompletionResponse{}, fmt.Errorf("decode modelslab response: %w", err)
	}
	if strings.EqualFold(strings.TrimSpace(decoded.Status), "error") {
		if strings.TrimSpace(decoded.Message) != "" {
			return CompletionResponse{}, fmt.Errorf("modelslab API error: %s", decoded.Message)
		}
		if decoded.Error != nil && decoded.Error.Message != "" {
			return CompletionResponse{}, fmt.Errorf("modelslab API error: %s", decoded.Error.Message)
		}
		return CompletionResponse{}, fmt.Errorf("modelslab API error: unknown error response")
	}

	if len(decoded.Choices) == 0 {
		if strings.TrimSpace(decoded.Message) != "" {
			return CompletionResponse{}, fmt.Errorf("modelslab API error: %s", decoded.Message)
		}
		return CompletionResponse{}, fmt.Errorf("modelslab API returned no choices")
	}

	message := decoded.Choices[0].Message
	toolCalls := make([]ToolCall, 0, len(message.ToolCalls))
	for _, call := range message.ToolCalls {
		toolCalls = append(toolCalls, ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: call.Function.Arguments,
		})
	}

	return CompletionResponse{
		Text:      extractModelslabText(message.Content),
		ToolCalls: toolCalls,
	}, nil
}

func decodeModelsLabResponseError(body []byte, status string) error {
	var decoded modelsLabChatResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		if trimmed := strings.TrimSpace(string(body)); trimmed != "" {
			return fmt.Errorf("modelslab API error: %s", trimmed)
		}
		return fmt.Errorf("decode modelslab error response: %w", err)
	}
	if decoded.Error != nil && strings.TrimSpace(decoded.Error.Message) != "" {
		return fmt.Errorf("modelslab API error: %s", decoded.Error.Message)
	}
	if strings.TrimSpace(decoded.Message) != "" {
		return fmt.Errorf("modelslab API error: %s", decoded.Message)
	}
	return fmt.Errorf("modelslab API error: status %s", status)
}

const maxModelsLabResponseBytes int64 = 16 << 20

type maxBytesReader struct {
	reader    io.Reader
	remaining int64
}

func newMaxBytesReader(r io.Reader, limit int64) *maxBytesReader {
	return &maxBytesReader{
		reader:    r,
		remaining: limit,
	}
}

func (r *maxBytesReader) Read(p []byte) (int, error) {
	if r.remaining <= 0 {
		return 0, fmt.Errorf("modelslab response exceeded %d bytes", maxModelsLabResponseBytes)
	}
	if int64(len(p)) > r.remaining {
		p = p[:int(r.remaining)]
	}
	n, err := r.reader.Read(p)
	r.remaining -= int64(n)
	if r.remaining <= 0 && err == nil {
		return n, fmt.Errorf("modelslab response exceeded %d bytes", maxModelsLabResponseBytes)
	}
	return n, err
}

func extractModelslabText(content any) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []any:
		lines := []string{}
		for _, item := range typed {
			block, ok := item.(map[string]any)
			if !ok {
				continue
			}
			text, _ := block["text"].(string)
			if text != "" {
				lines = append(lines, text)
			}
		}
		return strings.Join(lines, "\n")
	default:
		return ""
	}
}

func nullableString(value string) any {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	return value
}
