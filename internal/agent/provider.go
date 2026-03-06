package agent

import "context"

type Client interface {
	Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error)
}

func NewClient(token string) Client {
	return &modelsLabClient{
		token: token,
	}
}
