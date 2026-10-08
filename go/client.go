package von

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Model is the Von model version this SDK targets. Von ships exactly one model
// and the server always stamps responses with the version it actually served,
// so this is a request hint rather than a selector.
const Model = "von-1.3.0"

const (
	defaultBaseURL = "http://localhost:8000"
	defaultTimeout = 30 * time.Second

	defaultDecideInstructions = "Which option best describes the state?"
	defaultRateInstructions   = "Rate the severity or level:"
)

// ClientOptions configures a VonClient. Empty fields fall back to the
// VON_BASE_URL / TYPESAFE_BASE_URL and VON_API_KEY / TYPESAFE_API_KEY
// environment variables, then to http://localhost:8000 with no auth.
type ClientOptions struct {
	// BaseURL is the Von server root, e.g. http://localhost:8000.
	BaseURL string
	// APIKey, when set, is sent as "Authorization: Bearer <key>".
	APIKey string
	// Timeout bounds each request. Defaults to 30s.
	Timeout time.Duration
	// HTTPClient overrides the underlying client. When nil, a client with
	// Timeout is created.
	HTTPClient *http.Client
}

// VonClient is an HTTP client for a Von server.
type VonClient struct {
	BaseURL string
	APIKey  string
	Timeout time.Duration

	http *http.Client
}

// TypeSafeClient is a drop-in alias for VonClient, mirroring the Python and
// TypeScript SDKs' migration path from TypeSafe Jev.
type TypeSafeClient = VonClient

// NewClient returns a client configured from opts and the environment.
func NewClient(opts ClientOptions) *VonClient {
	base := firstNonEmpty(
		opts.BaseURL,
		os.Getenv("VON_BASE_URL"),
		os.Getenv("TYPESAFE_BASE_URL"),
		defaultBaseURL,
	)
	key := firstNonEmpty(
		opts.APIKey,
		os.Getenv("VON_API_KEY"),
		os.Getenv("TYPESAFE_API_KEY"),
	)
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	hc := opts.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: timeout}
	}
	return &VonClient{
		BaseURL: strings.TrimRight(base, "/"),
		APIKey:  key,
		Timeout: timeout,
		http:    hc,
	}
}

// NewTypeSafeClient is NewClient under the migration alias.
func NewTypeSafeClient(opts ClientOptions) *VonClient { return NewClient(opts) }

type systemOneRequest struct {
	Model     string              `json:"model"`
	State     any                 `json:"state"`
	Questions map[string]Question `json:"questions"`
}

// SystemOne evaluates a state and questions in a single forward pass
// (POST /v1/systemone) using the default model hint.
func (c *VonClient) SystemOne(ctx context.Context, state any, questions map[string]Question) (*SystemOneResponse, error) {
	return c.SystemOneModel(ctx, state, questions, Model)
}

// SystemOneModel is SystemOne with an explicit model hint.
func (c *VonClient) SystemOneModel(ctx context.Context, state any, questions map[string]Question, model string) (*SystemOneResponse, error) {
	if len(questions) == 0 {
		return nil, &VonError{Message: "von: at least one question is required"}
	}
	body, err := json.Marshal(systemOneRequest{Model: model, State: state, Questions: questions})
	if err != nil {
		return nil, &VonError{Message: "von: encode request: " + err.Error()}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, &VonError{Message: "von: build request: " + err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || isTimeout(err) {
			return nil, &VonError{Message: fmt.Sprintf("von: request timed out after %s", c.Timeout)}
		}
		return nil, &VonError{Message: "von: network failure: " + err.Error()}
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &VonError{Message: "von: read response: " + err.Error()}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &VonError{
			Message: fmt.Sprintf("von: server error (%d): %s", resp.StatusCode, strings.TrimSpace(string(data))),
			Status:  resp.StatusCode,
			Details: string(data),
		}
	}

	var out SystemOneResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, &VonError{
			Message: "von: decode response: " + err.Error(),
			Status:  resp.StatusCode,
			Details: string(data),
		}
	}
	return &out, nil
}

// Decide is the discrete-choice helper: one Choice question answered under the
// "decision" id. The returned answer's Choice is the winning key.
func (c *VonClient) Decide(ctx context.Context, state any, criteria map[string]*string, instructions ...string) (*Answer, error) {
	return c.decide(ctx, state, Choice(instructionOr(instructions, defaultDecideInstructions), criteria))
}

// DecideOptions is Decide for bare option keys (nil descriptions).
func (c *VonClient) DecideOptions(ctx context.Context, state any, keys []string, instructions ...string) (*Answer, error) {
	return c.decide(ctx, state, ChoiceOptions(instructionOr(instructions, defaultDecideInstructions), keys...))
}

func (c *VonClient) decide(ctx context.Context, state any, q *ChoiceQuestion) (*Answer, error) {
	resp, err := c.SystemOne(ctx, state, map[string]Question{"decision": q})
	if err != nil {
		return nil, err
	}
	ans, ok := resp.Answers["decision"]
	if !ok {
		return nil, &VonError{Message: "von: response is missing the 'decision' answer"}
	}
	return &ans, nil
}

// Judge is the binary-verification helper: one Noul question answered under the
// "verdict" id. It returns the committed P(true) with the band rule applied.
// Pass nil criteria to use the backend's generic descriptions. To gate on the
// pre-band probability, read NoulRaw from a SystemOne response instead.
func (c *VonClient) Judge(ctx context.Context, state any, instructions string, criteria *NoulCriteria) (float64, error) {
	resp, err := c.SystemOne(ctx, state, map[string]Question{"verdict": Noul(instructions, criteria)})
	if err != nil {
		return 0, err
	}
	ans, ok := resp.Answers["verdict"]
	if !ok {
		return 0, &VonError{Message: "von: response is missing the 'verdict' answer"}
	}
	return ans.Noul, nil
}

// Rate is the ordinal helper: one Score question answered under the "rating"
// id. levels are ordered lowest first.
func (c *VonClient) Rate(ctx context.Context, state any, levels []string, instructions ...string) (*Answer, error) {
	return c.rate(ctx, state, Score(instructionOr(instructions, defaultRateInstructions), levels))
}

// RateDescriptions is Rate for level -> description criteria.
func (c *VonClient) RateDescriptions(ctx context.Context, state any, levels map[string]string, instructions ...string) (*Answer, error) {
	return c.rate(ctx, state, ScoreDescriptions(instructionOr(instructions, defaultRateInstructions), levels))
}

func (c *VonClient) rate(ctx context.Context, state any, q *ScoreQuestion) (*Answer, error) {
	resp, err := c.SystemOne(ctx, state, map[string]Question{"rating": q})
	if err != nil {
		return nil, err
	}
	ans, ok := resp.Answers["rating"]
	if !ok {
		return nil, &VonError{Message: "von: response is missing the 'rating' answer"}
	}
	return &ans, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func instructionOr(instructions []string, def string) string {
	if len(instructions) > 0 && instructions[0] != "" {
		return instructions[0]
	}
	return def
}

func isTimeout(err error) bool {
	var te interface{ Timeout() bool }
	return errors.As(err, &te) && te.Timeout()
}
