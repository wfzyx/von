package von

// Choice builds a discrete categorical question from option key to description.
// Pass a nil description to let the key stand in for its own description.
func Choice(instructions string, criteria map[string]*string) *ChoiceQuestion {
	if criteria == nil {
		criteria = map[string]*string{}
	}
	return &ChoiceQuestion{Instructions: instructions, Criteria: criteria}
}

// ChoiceOptions builds a Choice question from bare option keys, each with a nil
// description (the key stands in). It is the Go form of the JavaScript
// `choice(instructions, string[])` overload.
func ChoiceOptions(instructions string, keys ...string) *ChoiceQuestion {
	return Choice(instructions, Options(keys...))
}

// Noul builds a binary verification question. criteria may be nil, in which
// case the backend uses its generic "condition holds / is false" descriptions,
// which is the weakest path; prefer passing explicit criteria.
func Noul(instructions string, criteria *NoulCriteria) *NoulQuestion {
	return &NoulQuestion{Instructions: instructions, Criteria: criteria}
}

// Score builds an ordinal question from an ordered list of level names, lowest
// first.
func Score(instructions string, levels []string) *ScoreQuestion {
	return &ScoreQuestion{
		Instructions: instructions,
		Criteria:     ScoreCriteria{Levels: levels},
	}
}

// ScoreDescriptions builds an ordinal question from level to description.
func ScoreDescriptions(instructions string, levels map[string]string) *ScoreQuestion {
	return &ScoreQuestion{
		Instructions: instructions,
		Criteria:     ScoreCriteria{Descriptions: levels},
	}
}

// Options builds Choice criteria from bare option keys (nil descriptions).
func Options(keys ...string) map[string]*string {
	out := make(map[string]*string, len(keys))
	for _, k := range keys {
		out[k] = nil
	}
	return out
}

// Descriptions builds Choice criteria from option key to description.
func Descriptions(m map[string]string) map[string]*string {
	out := make(map[string]*string, len(m))
	for k, v := range m {
		desc := v
		out[k] = &desc
	}
	return out
}
