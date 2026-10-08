package von

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestChoiceBuildsCriteria(t *testing.T) {
	q := Choice("Select department", Descriptions(map[string]string{
		"billing": "Invoices",
		"tech":    "Bugs",
	}))
	if q.questionType() != QuestionChoice {
		t.Fatalf("questionType = %q", q.questionType())
	}
	if q.Instructions != "Select department" {
		t.Fatalf("instructions = %q", q.Instructions)
	}
	if len(q.Criteria) != 2 {
		t.Fatalf("criteria len = %d", len(q.Criteria))
	}
	if got := deref(q.Criteria["billing"]); got != "Invoices" {
		t.Fatalf("billing = %q", got)
	}
	if got := deref(q.Criteria["tech"]); got != "Bugs" {
		t.Fatalf("tech = %q", got)
	}
}

func TestChoiceOptionsNormalizesToNull(t *testing.T) {
	q := ChoiceOptions("Select option", "opt_a", "opt_b")
	if len(q.Criteria) != 2 {
		t.Fatalf("criteria len = %d", len(q.Criteria))
	}
	if q.Criteria["opt_a"] != nil || q.Criteria["opt_b"] != nil {
		t.Fatalf("expected nil descriptions, got %v", q.Criteria)
	}
}

func TestNoulBuildsCriteria(t *testing.T) {
	q := Noul("Is database down?", &NoulCriteria{True: "Down", False: "Up"})
	if q.questionType() != QuestionNoul {
		t.Fatalf("questionType = %q", q.questionType())
	}
	if q.Criteria == nil || q.Criteria.True != "Down" || q.Criteria.False != "Up" {
		t.Fatalf("criteria = %+v", q.Criteria)
	}
}

func TestScoreBuildsLevels(t *testing.T) {
	q := Score("Rate severity", []string{"Low", "Medium", "High"})
	if q.questionType() != QuestionScore {
		t.Fatalf("questionType = %q", q.questionType())
	}
	if !reflect.DeepEqual(q.Criteria.Levels, []string{"Low", "Medium", "High"}) {
		t.Fatalf("levels = %v", q.Criteria.Levels)
	}
}

func TestQuestionWireFormat(t *testing.T) {
	tests := []struct {
		name string
		q    Question
		want map[string]any
	}{
		{
			name: "choice",
			q:    Choice("Pick", Descriptions(map[string]string{"a": "A"})),
			want: map[string]any{
				"type":         "choice",
				"instructions": "Pick",
				"criteria":     map[string]any{"a": "A"},
			},
		},
		{
			name: "noul keys are true/false",
			q:    Noul("Holds?", &NoulCriteria{True: "yes", False: "no"}),
			want: map[string]any{
				"type":         "noul",
				"instructions": "Holds?",
				"criteria":     map[string]any{"true": "yes", "false": "no"},
			},
		},
		{
			name: "noul nil criteria serialises null",
			q:    Noul("Holds?", nil),
			want: map[string]any{
				"type":         "noul",
				"instructions": "Holds?",
				"criteria":     nil,
			},
		},
		{
			name: "score list",
			q:    Score("Rate", []string{"low", "high"}),
			want: map[string]any{
				"type":         "score",
				"instructions": "Rate",
				"criteria":     []any{"low", "high"},
			},
		},
		{
			name: "score descriptions",
			q:    ScoreDescriptions("Rate", map[string]string{"1": "low", "2": "high"}),
			want: map[string]any{
				"type":         "score",
				"instructions": "Rate",
				"criteria":     map[string]any{"1": "low", "2": "high"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.q)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("wire format\n got: %#v\nwant: %#v", got, tt.want)
			}
		})
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
