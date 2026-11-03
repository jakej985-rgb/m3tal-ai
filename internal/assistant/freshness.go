package assistant

import (
	"regexp"
	"strings"
	"time"
)

var datedRevision = regexp.MustCompile(`(?im)(?:last[\s_-]*(?:updated|reviewed|verified)|updated|reviewed|verified|published|date)\s*[:=]\s*["']?(\d{4}-\d{2}-\d{2})`)

var timeSensitiveTerms = []string{
	"install", "download", "release", "version", "port", "default", "api", "token",
	"supported", "compatibility", "latest", "command", "package", "apt", "required",
}

// FreshnessAssessment flags local evidence that documentation may be outdated.
// It does not independently verify claims against external facts.
type FreshnessAssessment struct {
	Label  string
	Detail string
}

// AssessFreshness checks explicit update dates and detects changeable setup or
// version details that lack a review date.
func AssessFreshness(text string, now time.Time) FreshnessAssessment {
	var latest time.Time
	for _, match := range datedRevision.FindAllStringSubmatch(text, -1) {
		date, err := time.Parse("2006-01-02", match[1])
		if err == nil && date.After(latest) {
			latest = date
		}
	}
	if !latest.IsZero() {
		if latest.After(now.AddDate(0, 0, 1)) {
			return FreshnessAssessment{Label: "Review date is in the future", Detail: latest.Format("2006-01-02")}
		}
		if latest.Before(now.AddDate(-1, -6, 0)) {
			return FreshnessAssessment{Label: "Potentially outdated", Detail: "last listed date " + latest.Format("2006-01-02") + " is over 18 months old"}
		}
		return FreshnessAssessment{Label: "Recent date, unverified", Detail: "latest listed date " + latest.Format("2006-01-02") + "; content was not independently verified"}
	}

	for _, term := range timeSensitiveTerms {
		if containsTerm(strings.ToLower(text), term) {
			return FreshnessAssessment{Label: "Review date missing", Detail: "contains setup, version, or compatibility details without a listed review date"}
		}
	}
	return FreshnessAssessment{Label: "Date unknown", Detail: "no listed update or review date found"}
}

func containsTerm(text, term string) bool {
	for _, word := range features(text) {
		if word == term {
			return true
		}
	}
	return false
}
