package von

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTypeSafeAlias(t *testing.T) {
	c := NewClient(ClientOptions{BaseURL: "http://localhost:8000"})
	var alias *TypeSafeClient = c
	if alias.BaseURL != "http://localhost:8000" {
		t.Fatalf("baseURL = %q", alias.BaseURL)
	}
	c2 := NewTypeSafeClient(ClientOptions{BaseURL: "http://example.test/"})
	if c2.BaseURL != "http://example.test" {
		t.Fatalf("trailing slash not trimmed: %q", c2.BaseURL)
	}
}

func TestDefaults(t *testing.T) {
	t.Setenv("VON_BASE_URL", "")
	t.Setenv("TYPESAFE_BASE_URL", "")
	t.Setenv("VON_API_KEY", "")
	t.Setenv("TYPESAFE_API_KEY", "")
	c := NewClient(ClientOptions{})
	if c.BaseURL != "http://localhost:8000" {
		t.Fatalf("baseURL = %q", c.BaseURL)
	}
	if c.Timeout <= 0 {
		t.Fatalf("timeout = %v", c.Timeout)
	}
}

func TestEnvFallback(t *testing.T) {
	t.Setenv("VON_BASE_URL", "http://env.test:9999")
	t.Setenv("TYPESAFE_API_KEY", "envkey")
	c := NewClient(ClientOptions{})
	if c.BaseURL != "http://env.test:9999" {
		t.Fatalf("baseURL = %q", c.BaseURL)
	}
	if c.APIKey != "envkey" {
		t.Fatalf("apiKey = %q", c.APIKey)
	}
}

func TestExplicitOptionBeatsEnv(t *testing.T) {
	t.Setenv("VON_BASE_URL", "http://env.test:9999")
	c := NewClient(ClientOptions{BaseURL: "http://explicit.test", APIKey: "explicit"})
	if c.BaseURL != "http://explicit.test" || c.APIKey != "explicit" {
		t.Fatalf("opts not honoured: %q %q", c.BaseURL, c.APIKey)
	}
}

func TestSystemOnePayload(t *testing.T) {
	var gotPath, gotAuth, gotContentType string
	var gotBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model": "von-1.3.0",
			"answers": {
				"dept": {
					"type": "choice",
					"choice": "billing",
					"confidence": 0.92,
					"probabilities": {"billing": 0.92, "tech": 0.08}
				}
			},
			"usage": {"input_tokens": 10, "output_tokens": 8}
		}`)
	}))
	defer srv.Close()

	c := NewClient(ClientOptions{BaseURL: srv.URL, APIKey: "test_token"})
	resp, err := c.SystemOne(context.Background(), "Refund requested", map[string]Question{
		"dept": Choice("Department", Descriptions(map[string]string{"billing": "billing", "tech": "tech"})),
	})
	if err != nil {
		t.Fatalf("SystemOne: %v", err)
	}

	if gotPath != "/v1/systemone" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAuth != "Bearer test_token" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Fatalf("content-type = %q", gotContentType)
	}
	if gotBody["model"] != "von-1.3.0" {
		t.Fatalf("model = %v", gotBody["model"])
	}
	if gotBody["state"] != "Refund requested" {
		t.Fatalf("state = %v", gotBody["state"])
	}
	questions, ok := gotBody["questions"].(map[string]any)
	if !ok {
		t.Fatalf("questions = %#v", gotBody["questions"])
	}
	dept, ok := questions["dept"].(map[string]any)
	if !ok {
		t.Fatalf("questions.dept = %#v", questions["dept"])
	}
	if dept["type"] != "choice" {
		t.Fatalf("questions.dept.type = %v", dept["type"])
	}
	if dept["instructions"] != "Department" {
		t.Fatalf("questions.dept.instructions = %v", dept["instructions"])
	}

	if resp.Model != "von-1.3.0" {
		t.Fatalf("resp model = %q", resp.Model)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 8 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	ans, ok := resp.Answers["dept"]
	if !ok {
		t.Fatalf("missing dept answer")
	}
	if ans.Type != QuestionChoice || ans.Choice != "billing" || ans.Confidence != 0.92 {
		t.Fatalf("answer = %+v", ans)
	}
}

func TestDecideHelper(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		questions := body["questions"].(map[string]any)
		decision := questions["decision"].(map[string]any)
		if decision["instructions"] != "Which option best describes the state?" {
			t.Errorf("default instructions = %v", decision["instructions"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model": "von-1.3.0",
			"answers": {
				"decision": {
					"type": "choice",
					"choice": "refund",
					"confidence": 0.88,
					"probabilities": {"refund": 0.88, "tech": 0.12}
				}
			},
			"usage": {"input_tokens": 15, "output_tokens": 8}
		}`)
	}))
	defer srv.Close()

	c := NewClient(ClientOptions{BaseURL: srv.URL})
	ans, err := c.Decide(context.Background(), "Charge me twice!", Options("refund", "tech"))
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if ans.Choice != "refund" || ans.Confidence != 0.88 {
		t.Fatalf("answer = %+v", ans)
	}
}

func TestJudgeHelper(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		questions := body["questions"].(map[string]any)
		verdict := questions["verdict"].(map[string]any)
		criteria := verdict["criteria"].(map[string]any)
		if criteria["true"] != "Down" || criteria["false"] != "Up" {
			t.Errorf("criteria = %v", criteria)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model": "von-1.3.0",
			"answers": {"verdict": {"type": "noul", "noul": 0.9412, "noul_raw": 0.62}},
			"usage": {"input_tokens": 12, "output_tokens": 2}
		}`)
	}))
	defer srv.Close()

	c := NewClient(ClientOptions{BaseURL: srv.URL})
	got, err := c.Judge(context.Background(), "Pool exhausted", "Is the DB down?", &NoulCriteria{True: "Down", False: "Up"})
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got != 0.9412 {
		t.Fatalf("noul = %v", got)
	}
}

func TestRateHelper(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model": "von-1.3.0",
			"answers": {
				"rating": {
					"type": "score",
					"score": 2.4,
					"confidence": 0.7,
					"legend": {"1": "Low", "2": "Medium", "3": "High"},
					"probabilities": {"1": 0.1, "2": 0.4, "3": 0.5}
				}
			},
			"usage": {"input_tokens": 9, "output_tokens": 3}
		}`)
	}))
	defer srv.Close()

	c := NewClient(ClientOptions{BaseURL: srv.URL})
	ans, err := c.Rate(context.Background(), "Memory at 98%", []string{"Low", "Medium", "High"})
	if err != nil {
		t.Fatalf("Rate: %v", err)
	}
	if ans.Type != QuestionScore || ans.Score != 2.4 || ans.Legend["3"] != "High" {
		t.Fatalf("answer = %+v", ans)
	}
}

func TestServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"detail":"boom"}`)
	}))
	defer srv.Close()

	c := NewClient(ClientOptions{BaseURL: srv.URL})
	_, err := c.SystemOne(context.Background(), "x", map[string]Question{
		"q": ChoiceOptions("Pick", "a", "b"),
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	var ve *VonError
	if !errors.As(err, &ve) {
		t.Fatalf("error type = %T", err)
	}
	if ve.Status != http.StatusInternalServerError {
		t.Fatalf("status = %d", ve.Status)
	}
	if ve.Details != `{"detail":"boom"}` {
		t.Fatalf("details = %q", ve.Details)
	}
}

func TestNoQuestions(t *testing.T) {
	c := NewClient(ClientOptions{BaseURL: "http://localhost:8000"})
	_, err := c.SystemOne(context.Background(), "x", nil)
	var ve *VonError
	if !errors.As(err, &ve) {
		t.Fatalf("error type = %T", err)
	}
}
