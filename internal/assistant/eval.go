package assistant

import (
	"fmt"
	"strings"
)

// EvalQuestion is a test question with expected answer source
type EvalQuestion struct {
	Question         string
	ExpectedSource   string // file path
	ExpectedHeading  string // section heading
	IsDocsQuestion   bool   // true if this is a docs question, false if Core/query
}

// EvalSet is a collection of evaluation questions for testing retrieval
var EvalSet = []EvalQuestion{
	{
		Question:        "how do I install m3tal",
		ExpectedSource:  "M3tal-Hub",
		ExpectedHeading: "Install",
		IsDocsQuestion:  true,
	},
	{
		Question:        "where is the m3tal config file",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Configuration",
		IsDocsQuestion:  true,
	},
	{
		Question:        "explain the traefik setup",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Traefik",
		IsDocsQuestion:  true,
	},
	{
		Question:        "what apps are in m3tal hub",
		ExpectedSource:  "M3tal-Hub",
		ExpectedHeading: "Applications",
		IsDocsQuestion:  true,
	},
	{
		Question:        "how does m3tal core work",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Architecture",
		IsDocsQuestion:  true,
	},
	{
		Question:        "how do I configure the api token",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "API",
		IsDocsQuestion:  true,
	},
	{
		Question:        "where can I find the documentation",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Documentation",
		IsDocsQuestion:  true,
	},
	{
		Question:        "help me set up m3tal",
		ExpectedSource:  "M3tal-Hub",
		ExpectedHeading: "Setup",
		IsDocsQuestion:  true,
	},
	{
		Question:        "how do I expose an app through the proxy",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Proxy",
		IsDocsQuestion:  true,
	},
	{
		Question:        "what does the doctor command do",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Doctor",
		IsDocsQuestion:  true,
	},
	{
		Question:        "where are docker compose files stored",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Docker",
		IsDocsQuestion:  true,
	},
	{
		Question:        "how does docker work in m3tal",
		ExpectedSource:  "M3tal-core",
		ExpectedHeading: "Docker",
		IsDocsQuestion:  true,
	},
}

// EvalResult stores metrics for a single question
type EvalResult struct {
	Question            string
	RetrievedSource     string
	RetrievedHeading    string
	SourceMatch         bool
	HasRetrieval        bool
	HasRAGAnswer        bool
	RAGCitesSource      bool
	RAGValidUTF8        bool
	RAGSaysDontKnow     bool
	CitedSources        []string
}

// EvalMetrics aggregates results across the eval set
type EvalMetrics struct {
	TotalQuestions       int
	RetrievalRate        float64 // fraction that returned any results
	SourceMatchRate      float64 // fraction where expected source was in top results
	RAGRate              float64 // fraction where LM generated an answer
	RAGCitationRate      float64 // fraction of RAG answers that cite a source
	RAGValidUTF8Rate     float64 // fraction of RAG answers that are valid UTF-8
	RAGDontKnowRate      float64 // fraction where RAG says "I don't know" appropriately
	Results              []EvalResult
}

// RunEval tests retrieval and RAG on the eval set
func (assistant *Assistant) RunEval() EvalMetrics {
	metrics := EvalMetrics{Results: make([]EvalResult, 0, len(EvalSet))}
	
	for _, q := range EvalSet {
		result := EvalResult{Question: q.Question}
		
		if q.IsDocsQuestion {
			results := assistant.knowledge.Search(q.Question, 3)
			result.HasRetrieval = len(results) > 0
			
			if len(results) > 0 {
				top := results[0]
				result.RetrievedSource = top.Source
				result.RetrievedHeading = top.Title
				
				// Check if expected source is in top results
				for _, r := range results {
					if strings.Contains(strings.ToLower(r.Source), strings.ToLower(q.ExpectedSource)) {
						result.SourceMatch = true
						break
					}
				}
				
				// Test RAG if LM is available
				if assistant.HasLanguageModel() {
					draft, err := assistant.draftKnowledgeAnswer(q.Question, results)
					if err == nil {
						result.HasRAGAnswer = true
						result.RAGCitesSource = draftCitesSources(draft, results)
						for _, c := range extractCitations(draft, results) {
							result.CitedSources = append(result.CitedSources, c.Source)
						}
						result.RAGValidUTF8 = strings.ToValidUTF8(draft, "") == draft
						result.RAGSaysDontKnow = draftStatesUncertainty(draft)
					}
				}
			}
		}
		
		metrics.Results = append(metrics.Results, result)
	}
	
	// Compute aggregate metrics
	metrics.TotalQuestions = len(EvalSet)
	
	withRetrieval := 0
	withSourceMatch := 0
	withRAG := 0
	withCitation := 0
	withValidUTF8 := 0
	withDontKnow := 0
	
	for _, r := range metrics.Results {
		if r.HasRetrieval {
			withRetrieval++
		}
		if r.SourceMatch {
			withSourceMatch++
		}
		if r.HasRAGAnswer {
			withRAG++
			if r.RAGCitesSource {
				withCitation++
			}
			if r.RAGValidUTF8 {
				withValidUTF8++
			}
			if r.RAGSaysDontKnow {
				withDontKnow++
			}
		}
	}
	
	if metrics.TotalQuestions > 0 {
		metrics.RetrievalRate = float64(withRetrieval) / float64(metrics.TotalQuestions)
		metrics.SourceMatchRate = float64(withSourceMatch) / float64(metrics.TotalQuestions)
		metrics.RAGRate = float64(withRAG) / float64(metrics.TotalQuestions)
	}
	
	if withRAG > 0 {
		metrics.RAGCitationRate = float64(withCitation) / float64(withRAG)
		metrics.RAGValidUTF8Rate = float64(withValidUTF8) / float64(withRAG)
		metrics.RAGDontKnowRate = float64(withDontKnow) / float64(withRAG)
	}
	
	return metrics
}

// PlanEvalResult stores metrics for plan validity testing
type PlanEvalResult struct {
	Prompt       string
	ValidPlan    bool
	ParseError   string
	ExecutedSteps int
}

// PlanEvalMetrics aggregates plan validity results
type PlanEvalMetrics struct {
	TotalPrompts    int
	ValidPlanRate   float64
	Results         []PlanEvalResult
}

// MultiStepPrompts are test prompts for plan generation
var MultiStepPrompts = []string{
	"show stats and health",
	"show stats then check containers",
	"show stats and health and containers",
	"restart stack media and show logs",
	"start container jellyfin and stop container traefik",
}

// RunPlanEval tests how often the LM generates valid plans
func (assistant *Assistant) RunPlanEval() PlanEvalMetrics {
	metrics := PlanEvalMetrics{Results: make([]PlanEvalResult, 0, len(MultiStepPrompts))}
	
	for _, prompt := range MultiStepPrompts {
		result := PlanEvalResult{Prompt: prompt}
		
		if assistant.languageModel != nil && looksMultiStep(prompt) {
			plan, err := assistant.proposePlan(prompt)
			if err == nil {
				result.ValidPlan = true
				result.ExecutedSteps = len(plan.Steps)
			} else {
				result.ParseError = err.Error()
			}
		}
		
		metrics.Results = append(metrics.Results, result)
	}
	
	metrics.TotalPrompts = len(MultiStepPrompts)
	validCount := 0
	for _, r := range metrics.Results {
		if r.ValidPlan {
			validCount++
		}
	}
	if metrics.TotalPrompts > 0 {
		metrics.ValidPlanRate = float64(validCount) / float64(metrics.TotalPrompts)
	}
	
	return metrics
}

// PrintEvalReport writes a human-readable eval report
func PrintEvalReport(metrics EvalMetrics) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Eval Report (%d questions)\n", metrics.TotalQuestions)
	fmt.Fprintf(&b, "Retrieval rate: %.1f%% (%d/%d returned results)\n", 
		metrics.RetrievalRate*100, int(metrics.RetrievalRate*float64(metrics.TotalQuestions)), metrics.TotalQuestions)
	fmt.Fprintf(&b, "Source match rate: %.1f%% (%d/%d matched expected source)\n", 
		metrics.SourceMatchRate*100, int(metrics.SourceMatchRate*float64(metrics.TotalQuestions)), metrics.TotalQuestions)
	fmt.Fprintf(&b, "RAG generation rate: %.1f%% (%d/%d generated answers)\n", 
		metrics.RAGRate*100, int(metrics.RAGRate*float64(metrics.TotalQuestions)), metrics.TotalQuestions)
	if metrics.RAGRate > 0 {
		fmt.Fprintf(&b, "RAG citation rate: %.1f%% (%d/%d cited sources)\n", 
			metrics.RAGCitationRate*100, int(metrics.RAGCitationRate*float64(metrics.TotalQuestions)), int(metrics.RAGRate*float64(metrics.TotalQuestions)))
		fmt.Fprintf(&b, "RAG valid UTF-8 rate: %.1f%% (%d/%d valid UTF-8)\n", 
			metrics.RAGValidUTF8Rate*100, int(metrics.RAGValidUTF8Rate*float64(metrics.TotalQuestions)), int(metrics.RAGRate*float64(metrics.TotalQuestions)))
		fmt.Fprintf(&b, "RAG 'don't know' rate: %.1f%% (%d/%d appropriately uncertain)\n", 
			metrics.RAGDontKnowRate*100, int(metrics.RAGDontKnowRate*float64(metrics.TotalQuestions)), int(metrics.RAGRate*float64(metrics.TotalQuestions)))
	}
	
	fmt.Fprintln(&b, "\nPer-question results:")
	for _, r := range metrics.Results {
		fmt.Fprintf(&b, "- %s\n", r.Question)
		if r.HasRetrieval {
			fmt.Fprintf(&b, "  Retrieved: %s (%s)\n", r.RetrievedHeading, r.RetrievedSource)
			if r.SourceMatch {
				fmt.Fprintln(&b, "  ✓ Source match")
			} else {
				fmt.Fprintln(&b, "  ✗ Source mismatch")
			}
			if r.HasRAGAnswer {
				fmt.Fprintf(&b, "  RAG: cites=%v valid-utf8=%v dont-know=%v", 
					r.RAGCitesSource, r.RAGValidUTF8, r.RAGSaysDontKnow)
				if len(r.CitedSources) > 0 {
					fmt.Fprintf(&b, " cited=%v", r.CitedSources)
				}
				fmt.Fprintln(&b)
			}
		} else {
			fmt.Fprintln(&b, "  ✗ No retrieval")
		}
	}
	
	return b.String()
}

// PrintPlanEvalReport writes a human-readable plan eval report
func PrintPlanEvalReport(metrics PlanEvalMetrics) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Plan Eval Report (%d prompts)\n", metrics.TotalPrompts)
	fmt.Fprintf(&b, "Valid plan rate: %.1f%% (%d/%d)\n", 
		metrics.ValidPlanRate*100, int(metrics.ValidPlanRate*float64(metrics.TotalPrompts)), metrics.TotalPrompts)
	
	fmt.Fprintln(&b, "\nPer-prompt results:")
	for _, r := range metrics.Results {
		fmt.Fprintf(&b, "- %s\n", r.Prompt)
		if r.ValidPlan {
			fmt.Fprintf(&b, "  ✓ Valid plan (%d steps)\n", r.ExecutedSteps)
		} else {
			fmt.Fprintf(&b, "  ✗ Invalid: %s\n", r.ParseError)
		}
	}
	
	return b.String()
}
