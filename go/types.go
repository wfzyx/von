// Package von is the Go client for Von, the open-source non-autoregressive
// System One decision model. It speaks the TypeSafe /v1/systemone wire protocol
// implemented by `von serve`.
//
// The SDK mirrors the Python (von-sdk) and TypeScript (von-sdk) clients: build
// questions, send them in a single forward pass, read the answers. Unlike the
// Python SDK it is HTTP-only; there is no in-process inference backend.
//
// Build questions with Choice, ChoiceOptions, Noul, Score or ScoreDescriptions
// rather than by hand, then send them with VonClient.SystemOne or one of the
// Decide / Judge / Rate helpers.
package von

import "encoding/json"

// QuestionType identifies the kind of decision a Question asks for.
type QuestionType string

const (
	// QuestionChoice asks for one of K described options.
	QuestionChoice QuestionType = "choice"
	// QuestionNoul asks for P(a condition holds).
	QuestionNoul QuestionType = "noul"
	// QuestionScore asks for a position on an ordinal scale.
	QuestionScore QuestionType = "score"
)

// Question is a single decision sent in the "questions" map of a System One
// request. Build one with Choice, ChoiceOptions, Noul, Score or
// ScoreDescriptions; the concrete types declare an unexported marker method, so
// only SDK-built pointers satisfy this interface.
type Question interface {
	questionType() QuestionType
}

// NoulCriteria constrains a Noul question with explicit positive and negative
// descriptions. The wire keys are "true" and "false" (the keys the Von backend
// reads); an empty field is omitted. Passing nil to Noul falls back to the
// backend's generic "holds / is false" descriptions.
type NoulCriteria struct {
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

// NoulQuestion is a binary verification question: P(a condition holds).
type NoulQuestion struct {
	Instructions string        `json:"instructions"`
	Criteria     *NoulCriteria `json:"criteria"`
}

func (*NoulQuestion) questionType() QuestionType { return QuestionNoul }

// MarshalJSON stamps the discriminator the server expects.
func (q *NoulQuestion) MarshalJSON() ([]byte, error) {
	type alias NoulQuestion
	return json.Marshal(struct {
		Type QuestionType `json:"type"`
		*alias
	}{QuestionNoul, (*alias)(q)})
}

// ChoiceQuestion picks one option from a fixed list and returns a probability
// distribution over all of them. A nil description means the option key stands
// in for its own description.
type ChoiceQuestion struct {
	Instructions string             `json:"instructions"`
	Criteria     map[string]*string `json:"criteria"`
}

func (*ChoiceQuestion) questionType() QuestionType { return QuestionChoice }

// MarshalJSON stamps the discriminator the server expects.
func (q *ChoiceQuestion) MarshalJSON() ([]byte, error) {
	type alias ChoiceQuestion
	return json.Marshal(struct {
		Type QuestionType `json:"type"`
		*alias
	}{QuestionChoice, (*alias)(q)})
}

// ScoreCriteria is either an ordered list of level names or a map from level to
// description. Build one with Score (levels) or ScoreDescriptions (map). The
// zero value serialises as an empty list.
type ScoreCriteria struct {
	Levels       []string
	Descriptions map[string]string
}

// MarshalJSON emits the list form or the map form, whichever was set.
func (s ScoreCriteria) MarshalJSON() ([]byte, error) {
	if s.Descriptions != nil {
		return json.Marshal(s.Descriptions)
	}
	if s.Levels == nil {
		return []byte("[]"), nil
	}
	return json.Marshal(s.Levels)
}

// ScoreQuestion places the state on an ordered scale and returns a
// probability-weighted position over the levels.
type ScoreQuestion struct {
	Instructions string        `json:"instructions"`
	Criteria     ScoreCriteria `json:"criteria"`
}

func (*ScoreQuestion) questionType() QuestionType { return QuestionScore }

// MarshalJSON stamps the discriminator the server expects.
func (q *ScoreQuestion) MarshalJSON() ([]byte, error) {
	type alias ScoreQuestion
	return json.Marshal(struct {
		Type QuestionType `json:"type"`
		*alias
	}{QuestionScore, (*alias)(q)})
}

// Answer is the response to a single question. Type names the primitive it came
// from; fields that do not apply are left zero.
type Answer struct {
	Type QuestionType `json:"type"`

	// Choice answers.
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`

	// Choice and Score answers.
	Confidence float64 `json:"confidence,omitempty"`

	// Noul answers. Noul is the committed decision after the band rule; NoulRaw
	// is the calibrated P(true) before the band rule and is the value to gate
	// on. NoulRaw is nil when the server did not send it.
	Noul    float64  `json:"noul,omitempty"`
	NoulRaw *float64 `json:"noul_raw,omitempty"`

	// Score answers.
	Score  float64           `json:"score,omitempty"`
	Legend map[string]string `json:"legend,omitempty"`
}

// Usage reports the tokenizer counts the server measured for the request.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// SystemOneResponse is the envelope returned by /v1/systemone.
type SystemOneResponse struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
	// Truncation is present only when the state was middle-truncated to fit the
	// encoder window or VON_MAX_STATE_TOKENS. It is an extra beyond the
	// TypeSafe spec; clients may ignore it.
	Truncation map[string]any `json:"truncation,omitempty"`
}
