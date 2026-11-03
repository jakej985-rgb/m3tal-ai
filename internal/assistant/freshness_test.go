package assistant

import (
	"testing"
	"time"
)

func TestAssessFreshnessFlagsOldAndUndatedOperationalGuidance(t *testing.T) {
	now := time.Date(2026, time.November, 3, 0, 0, 0, 0, time.UTC)
	cases := []struct {
		text  string
		label string
	}{
		{"Last reviewed: 2024-01-10\nInstall the latest package.", "Potentially outdated"},
		{"date: 2026-10-10\nInstall the package.", "Recent date, unverified"},
		{"Use apt to install the current version.", "Review date missing"},
		{"A neural network maps vectors to outputs.", "Date unknown"},
	}
	for _, test := range cases {
		got := AssessFreshness(test.text, now)
		if got.Label != test.label {
			t.Errorf("AssessFreshness(%q) label = %q, want %q", test.text, got.Label, test.label)
		}
	}
}

func TestAssessFreshnessFlagsFutureDates(t *testing.T) {
	now := time.Date(2026, time.November, 3, 0, 0, 0, 0, time.UTC)
	got := AssessFreshness("Last updated: 2027-01-01", now)
	if got.Label != "Review date is in the future" {
		t.Fatalf("future date label = %q", got.Label)
	}
}
