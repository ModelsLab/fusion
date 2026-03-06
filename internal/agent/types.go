package agent

import "time"

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema map[string]any
}

type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
}

type Session struct {
	Version      int       `json:"version"`
	ID           string    `json:"id"`
	Provider     string    `json:"provider"`
	Model        string    `json:"model"`
	CWD          string    `json:"cwd"`
	SystemPrompt string    `json:"system_prompt"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	Messages     []Message `json:"messages"`
}

type CompletionRequest struct {
	Model        string
	SystemPrompt string
	Messages     []Message
	Tools        []ToolDefinition
	MaxTokens    int
}

type CompletionResponse struct {
	Text      string
	ToolCalls []ToolCall
}
