package cli

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/ModelsLab/fusion/internal/modelslab"
	"github.com/spf13/cobra"
)

type modelslabLoginOptions struct {
	Model     string
	Timeout   time.Duration
	WebURL    string
	NoBrowser bool
}

type modelslabLoginPayload struct {
	APIKey string `json:"api_key"`
	Model  string `json:"model_id"`
	State  string `json:"state"`
}

func runModelsLabBrowserLogin(cmd *cobra.Command, opts modelslabLoginOptions) error {
	runtimeState, err := loadRuntime()
	if err != nil {
		return err
	}

	modelID := strings.TrimSpace(opts.Model)
	if modelID == "" {
		cfg, loadErr := runtimeState.Config.Load()
		if loadErr != nil {
			return loadErr
		}
		modelID = strings.TrimSpace(cfg.ModelsLab.Model)
	}
	if modelID == "" {
		modelID = strings.TrimSpace(os.Getenv(modelslab.ModelEnvVar))
	}
	if modelID == "" {
		modelID = strings.TrimSpace(modelslab.DefaultModelID)
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 3 * time.Minute
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("start login callback server: %w", err)
	}
	defer listener.Close()

	state, err := newLoginState()
	if err != nil {
		return err
	}

	callbackURL := "http://" + listener.Addr().String() + "/callback"
	loginURL, err := buildModelsLabLoginURL(modelslab.NormalizeURL(opts.WebURL, modelslab.WebBaseURL()), callbackURL, state, modelID)
	if err != nil {
		return err
	}

	resultCh := make(chan modelslabLoginPayload, 1)
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			payload, parseErr := parseModelsLabLoginPayload(r)
			if parseErr != nil {
				http.Error(w, parseErr.Error(), http.StatusBadRequest)
				return
			}
			if payload.State != state {
				http.Error(w, "state mismatch", http.StatusUnauthorized)
				return
			}
			if strings.TrimSpace(payload.APIKey) == "" {
				http.Error(w, "api_key is required", http.StatusBadRequest)
				return
			}
			if strings.TrimSpace(payload.Model) == "" {
				payload.Model = modelID
			}

			select {
			case resultCh <- payload:
			default:
			}

			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = io.WriteString(w, "<html><body><h1>Fusion login complete</h1><p>You can return to the terminal.</p></body></html>")
		}),
	}

	go func() {
		_ = server.Serve(listener)
	}()
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownCtx)
	}()

	cmd.Printf("Fusion login URL: %s\n", loginURL)
	if !opts.NoBrowser {
		if openErr := openBrowser(loginURL); openErr != nil {
			cmd.Printf("warning: could not open browser automatically: %v\n", openErr)
			cmd.Println("Open the URL above in a browser to continue.")
		}
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case payload := <-resultCh:
		if err := runtimeState.Config.SetModelsLabToken(strings.TrimSpace(payload.APIKey)); err != nil {
			return err
		}
		if err := runtimeState.Config.SetModelsLabModel(strings.TrimSpace(payload.Model)); err != nil {
			return err
		}

		cmd.Printf("Stored ModelsLab credentials in %s\n", runtimeState.Config.Path())
		cmd.Printf("Default model: %s\n", strings.TrimSpace(payload.Model))
		return nil
	case <-timer.C:
		return fmt.Errorf("timed out waiting for ModelsLab browser login after %s", timeout)
	}
}

func buildModelsLabLoginURL(webURL string, callbackURL string, state string, modelID string) (string, error) {
	base, err := url.Parse(modelslab.NormalizeURL(webURL, modelslab.WebBaseURL()))
	if err != nil {
		return "", fmt.Errorf("parse ModelsLab web URL: %w", err)
	}

	base.Path = strings.TrimRight(base.Path, "/") + modelslab.LoginPath
	query := base.Query()
	query.Set("callback_url", callbackURL)
	query.Set("state", state)
	query.Set("model_id", strings.TrimSpace(modelID))
	base.RawQuery = query.Encode()
	return base.String(), nil
}

func parseModelsLabLoginPayload(r *http.Request) (modelslabLoginPayload, error) {
	var payload modelslabLoginPayload

	switch {
	case strings.Contains(r.Header.Get("Content-Type"), "application/json"):
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			return modelslabLoginPayload{}, fmt.Errorf("decode JSON payload: %w", err)
		}
	case r.Method == http.MethodPost:
		if err := r.ParseForm(); err != nil {
			return modelslabLoginPayload{}, fmt.Errorf("parse form payload: %w", err)
		}
		payload.APIKey = r.Form.Get("api_key")
		payload.Model = r.Form.Get("model_id")
		payload.State = r.Form.Get("state")
	default:
		payload.APIKey = r.URL.Query().Get("api_key")
		payload.Model = r.URL.Query().Get("model_id")
		payload.State = r.URL.Query().Get("state")
	}

	payload.APIKey = strings.TrimSpace(payload.APIKey)
	payload.Model = strings.TrimSpace(payload.Model)
	payload.State = strings.TrimSpace(payload.State)
	if payload.State == "" {
		return modelslabLoginPayload{}, fmt.Errorf("state is required")
	}
	return payload, nil
}

func newLoginState() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generate login state: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

func openBrowser(target string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", target)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", target)
	default:
		cmd = exec.Command("xdg-open", target)
	}
	return cmd.Start()
}
