package baloot

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestResearchResultOperatorAuditDecodesTypedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/api/v2/research-results/LAF-ABCDEF1234/operator-audit" {
			t.Fatalf("path = %q", request.URL.Path)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"schemaVersion":1,
			"resultCode":"LAF-ABCDEF1234",
			"researchAuthority":{"schemaVersion":1,"backendId":"ortools_cp_sat_v2","backendVersion":"2.0.0","qualificationAuthority":"cp_sat","solverVersion":"9.14","bundleRevisionId":"CBR-SDK-V4","bundleHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","domainIrHashes":{"rules":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"loweringManifestHash":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","constructionModelHash":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","proofModelHash":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","graphHash":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","traceHash":"1111111111111111111111111111111111111111111111111111111111111111","proofOutcomeHash":"2222222222222222222222222222222222222222222222222222222222222222","sha256":"3333333333333333333333333333333333333333333333333333333333333333"},
			"runtimeCompatibility":{"schemaVersion":1,"status":"passed","engineVersion":"21.2.0","bundleHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","traceHash":"1111111111111111111111111111111111111111111111111111111111111111","evidenceHash":"4444444444444444444444444444444444444444444444444444444444444444","failureCodes":[],"sha256":"5555555555555555555555555555555555555555555555555555555555555555"},
			"executionReceipt":{"schemaVersion":1,"semanticResultHash":"6666666666666666666666666666666666666666666666666666666666666666","stageTimings":{"proof":0.25},"cpuSeconds":0.2,"peakMemoryBytes":1024,"conflicts":3,"branches":4,"cacheBytes":512,"completedVariants":1,"machine":{"platform":"test"},"sha256":"7777777777777777777777777777777777777777777777777777777777777777"}
		}`))
	}))
	defer server.Close()

	var audit ResearchOperatorAuditV1Response
	if err := New(server.URL).ResearchResultOperatorAudit(context.Background(), "LAF-ABCDEF1234", &audit); err != nil {
		t.Fatal(err)
	}
	if audit.ResearchAuthority.BackendID != "ortools_cp_sat_v2" || audit.RuntimeCompatibility.Status != "passed" {
		t.Fatalf("decoded audit = %#v", audit)
	}
	if audit.ExecutionReceipt.StageTimings["proof"] != 0.25 || audit.ExecutionReceipt.CompletedVariants != 1 {
		t.Fatalf("decoded receipt = %#v", audit.ExecutionReceipt)
	}
}

func TestResearchResultFiltersCoverV4SemanticFacets(t *testing.T) {
	exactStateAvailable := true
	minimumScore := 80.0
	filters := ResearchResultFilters{
		MinimumScore:               &minimumScore,
		ScenarioTemplate:           "SCN-SDK",
		ScenarioOrigin:             "exact_trace_prefix",
		ScenarioStatus:             "satisfied",
		RequestedProofMode:         "exact",
		AchievedProofMode:          "exact",
		ProofCompletion:            "exact_complete",
		ProofVisibility:            "complete_state",
		TargetTeam:                 "north_south",
		ContractOutcome:            "lost",
		ProjectKind:                "hundred_sequence",
		ProjectValue:               "100",
		ProjectState:               "awarded",
		WinningRootActionBand:      "1",
		ExactStateAvailable:        &exactStateAvailable,
		DecisionKind:               "play",
		ResearchBackend:            "ortools_cp_sat_v2",
		QualificationAuthority:     "cp_sat",
		RuntimeCompatibilityStatus: "passed",
		ConfigurationBundle:        "CBR-SDK-V4",
		ExactStateID:               "STATE-1234567890ABCDEF1234",
		ScenarioCategory:           "proof",
		VariantCollection:          "RUN-ABCDEFGHIJKL",
		GroupBy:                    "scenario_category",
	}
	query := filters.Values()
	want := map[string]string{
		"minimumScore":               "80",
		"scenarioTemplate":           "SCN-SDK",
		"scenarioOrigin":             "exact_trace_prefix",
		"scenarioStatus":             "satisfied",
		"requestedProofMode":         "exact",
		"achievedProofMode":          "exact",
		"proofCompletion":            "exact_complete",
		"proofVisibility":            "complete_state",
		"targetTeam":                 "north_south",
		"contractOutcome":            "lost",
		"projectKind":                "hundred_sequence",
		"projectValue":               "100",
		"projectState":               "awarded",
		"winningRootActionBand":      "1",
		"exactStateAvailable":        "true",
		"decisionKind":               "play",
		"researchBackend":            "ortools_cp_sat_v2",
		"qualificationAuthority":     "cp_sat",
		"runtimeCompatibilityStatus": "passed",
		"configurationBundle":        "CBR-SDK-V4",
		"exactStateId":               "STATE-1234567890ABCDEF1234",
		"scenarioCategory":           "proof",
		"variantCollection":          "RUN-ABCDEFGHIJKL",
		"groupBy":                    "scenario_category",
	}
	for key, expected := range want {
		if got := query.Get(key); got != expected {
			t.Errorf("%s = %q, want %q", key, got, expected)
		}
	}
}
