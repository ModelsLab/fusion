package cli

import (
	"fmt"

	"github.com/ModelsLab/fusion/internal/config"
	"github.com/ModelsLab/fusion/internal/kb"
)

type runtimeState struct {
	Config *config.Manager
	KB     *kb.Store
}

func loadRuntime() (*runtimeState, error) {
	manager, err := config.NewManager()
	if err != nil {
		return nil, fmt.Errorf("initialize config manager: %w", err)
	}

	store, err := kb.LoadDefault()
	if err != nil {
		return nil, fmt.Errorf("load embedded knowledge base: %w", err)
	}

	return &runtimeState{
		Config: manager,
		KB:     store,
	}, nil
}
