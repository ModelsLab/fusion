package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type Engine struct {
	client        Client
	registry      *ToolRegistry
	maxToolRounds int
	onToolCall    func(ToolCall)
	onToolResult  func(ToolCall, string, error)
}

func NewEngine(client Client, registry *ToolRegistry, maxToolRounds int) *Engine {
	if maxToolRounds <= 0 {
		maxToolRounds = 12
	}
	return &Engine{
		client:        client,
		registry:      registry,
		maxToolRounds: maxToolRounds,
	}
}

func (e *Engine) SetToolHooks(onCall func(ToolCall), onResult func(ToolCall, string, error)) {
	e.onToolCall = onCall
	e.onToolResult = onResult
}

func (e *Engine) RunTurn(ctx context.Context, session *Session, prompt string) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", fmt.Errorf("prompt is required")
	}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 30*time.Minute)
		defer cancel()
	}

	now := time.Now().UTC()
	session.Messages = append(session.Messages, Message{
		Role:      "user",
		Content:   prompt,
		CreatedAt: now,
	})

	for round := 0; round < e.maxToolRounds; round++ {
		response, err := e.client.Complete(ctx, CompletionRequest{
			Model:        session.Model,
			SystemPrompt: session.SystemPrompt,
			Messages:     session.Messages,
			Tools:        e.registry.Definitions(),
			MaxTokens:    4096,
		})
		if err != nil {
			return "", err
		}

		session.Messages = append(session.Messages, Message{
			Role:      "assistant",
			Content:   strings.TrimSpace(response.Text),
			ToolCalls: response.ToolCalls,
			CreatedAt: time.Now().UTC(),
		})

		if len(response.ToolCalls) == 0 {
			return strings.TrimSpace(response.Text), nil
		}

		for _, call := range response.ToolCalls {
			if e.onToolCall != nil {
				e.onToolCall(call)
			}

			output, execErr := e.registry.Execute(ctx, call)
			if execErr != nil {
				output = toolErrorPayload(execErr)
			}

			session.Messages = append(session.Messages, Message{
				Role:       "tool",
				Content:    output,
				ToolCallID: call.ID,
				ToolName:   call.Name,
				CreatedAt:  time.Now().UTC(),
			})

			if e.onToolResult != nil {
				e.onToolResult(call, output, execErr)
			}
		}
	}

	return "", fmt.Errorf("max tool rounds exceeded")
}

func toolErrorPayload(err error) string {
	payload := map[string]any{
		"error": err.Error(),
	}
	data, marshalErr := json.MarshalIndent(payload, "", "  ")
	if marshalErr != nil {
		return `{"error":"tool execution failed"}`
	}
	return string(data)
}
