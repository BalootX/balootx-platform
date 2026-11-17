package baloot

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
)

func TestResearchBundleSelectorsPreserveLegacyReads(t *testing.T) {
	requests := make([]*http.Request, 0, 4)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests = append(requests, request)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{}`))
	}))
	defer server.Close()

	client := New(server.URL)
	var response map[string]any
	ctx := context.Background()
	if err := client.DomainCapabilities(ctx, true, &response); err != nil {
		t.Fatal(err)
	}
	if err := client.DomainCapabilities(ctx, false, &response, "CBR-A"); err != nil {
		t.Fatal(err)
	}
	if err := client.RulePrimitives(ctx, &response); err != nil {
		t.Fatal(err)
	}
	if err := client.RulePrimitives(ctx, &response, "CBR-B"); err != nil {
		t.Fatal(err)
	}

	if len(requests) != 4 {
		t.Fatalf("got %d requests, want 4", len(requests))
	}
	if got := requests[0].URL.Query().Get("includePayloads"); got != "true" {
		t.Fatalf("legacy includePayloads = %q, want true", got)
	}
	if requests[0].URL.Query().Has("bundleRevisionId") {
		t.Fatal("legacy domain capabilities request unexpectedly selected a bundle")
	}
	if got := requests[1].URL.Query().Get("bundleRevisionId"); got != "CBR-A" {
		t.Fatalf("domain capabilities bundleRevisionId = %q, want CBR-A", got)
	}
	if requests[2].URL.Query().Has("bundleRevisionId") {
		t.Fatal("legacy rule primitives request unexpectedly selected a bundle")
	}
	if got := requests[3].URL.Query().Get("bundleRevisionId"); got != "CBR-B" {
		t.Fatalf("rule primitives bundleRevisionId = %q, want CBR-B", got)
	}

	if err := client.RulePrimitives(ctx, &response, "CBR-A", "CBR-B"); err == nil {
		t.Fatal("multiple bundle selectors must fail before making a request")
	}
}

func TestResearchRenderRequestsForwardObserverSeat(t *testing.T) {
	paths := make([]string, 0, 3)
	bodies := make([]map[string]any, 0, 3)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		paths = append(paths, request.URL.Path)
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{}`))
	}))
	defer server.Close()

	client := New(server.URL)
	request := ResearchRenderPrepareRequest{
		RevealPolicy: "after_completion",
		ExportPreset: "decision_clip",
		ObserverSeat: ResearchObserverWest,
	}
	var response map[string]any
	ctx := context.Background()
	if err := client.PrepareResearchResultRender(ctx, "LAF-ABC123", request, &response); err != nil {
		t.Fatal(err)
	}
	request.ObserverSeat = ResearchObserverEast
	if err := client.PrepareResearchDecisionRender(ctx, "LAF-ABC123", 7, request, &response); err != nil {
		t.Fatal(err)
	}
	request.ObserverSeat = ResearchObserverNorth
	if err := client.PrepareLaffaRender(ctx, "LAT-ABC123", request, &response); err != nil {
		t.Fatal(err)
	}

	wantPaths := []string{
		"/api/v2/research-results/LAF-ABC123:prepare-render",
		"/api/v2/research-results/LAF-ABC123/decisions/7:prepare-render",
		"/api/v2/laffat/LAT-ABC123:prepare-render",
	}
	for index, want := range wantPaths {
		if paths[index] != want {
			t.Fatalf("request %d path = %q, want %q", index, paths[index], want)
		}
	}
	wantSeats := []string{"west", "east", "north"}
	for index, want := range wantSeats {
		if bodies[index]["observerSeat"] != want {
			t.Fatalf("request %d observerSeat = %v, want %q", index, bodies[index]["observerSeat"], want)
		}
	}
}

func TestResearchRecipeMutationsGenerateOnlyAbsentIdempotencyKeys(t *testing.T) {
	requests := make([]*http.Request, 0, 6)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests = append(requests, request)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{}`))
	}))
	defer server.Close()

	client := New(server.URL)
	ctx := context.Background()
	var response map[string]any
	payload := map[string]any{"schemaVersion": 3}
	if err := client.CreateResearchRecipe(ctx, payload, &response); err != nil {
		t.Fatal(err)
	}
	if err := client.CreateResearchRecipe(ctx, payload, &response, "create-key"); err != nil {
		t.Fatal(err)
	}
	if err := client.CreateResearchRecipe(ctx, payload, &response, ""); err != nil {
		t.Fatal(err)
	}
	if err := client.RetireResearchRecipe(ctx, "RCP-ABC123", &response); err != nil {
		t.Fatal(err)
	}
	if err := client.RetireResearchRecipe(ctx, "RCP-ABC123", &response, "retire-key"); err != nil {
		t.Fatal(err)
	}
	if err := client.RetireResearchRecipe(ctx, "RCP-ABC123", &response, ""); err != nil {
		t.Fatal(err)
	}

	if len(requests) != 6 {
		t.Fatalf("got %d requests, want 6", len(requests))
	}
	hexKey := regexp.MustCompile(`^[0-9a-f]{32}$`)
	for _, index := range []int{0, 3} {
		if key := requests[index].Header.Get("Idempotency-Key"); !hexKey.MatchString(key) {
			t.Fatalf("request %d generated key = %q, want 32 lowercase hex characters", index, key)
		}
	}
	if got := requests[1].Header.Get("Idempotency-Key"); got != "create-key" {
		t.Fatalf("explicit create key = %q, want create-key", got)
	}
	if got := requests[4].Header.Get("Idempotency-Key"); got != "retire-key" {
		t.Fatalf("explicit retire key = %q, want retire-key", got)
	}
	for _, index := range []int{2, 5} {
		values, present := requests[index].Header["Idempotency-Key"]
		if !present || len(values) != 1 || values[0] != "" {
			t.Fatalf("request %d explicit blank key was not preserved: %#v", index, values)
		}
	}

	if err := client.CreateResearchRecipe(ctx, payload, &response, "a", "b"); err == nil {
		t.Fatal("multiple create idempotency keys must fail before making a request")
	}
	if err := client.RetireResearchRecipe(ctx, "RCP-ABC123", &response, "a", "b"); err == nil {
		t.Fatal("multiple retire idempotency keys must fail before making a request")
	}
	if len(requests) != 6 {
		t.Fatalf("invalid variadics made requests: got %d, want 6", len(requests))
	}
}

func TestResearchRecipeListCursorAndTypedPagePreserveLegacyCallers(t *testing.T) {
	requests := make([]*http.Request, 0, 3)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests = append(requests, request)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"items":[{"schemaVersion":3}],"nextCursor":"NEXT"}`))
	}))
	defer server.Close()

	client := New(server.URL)
	ctx := context.Background()
	var page ResearchRecipeListResponse
	if err := client.ResearchRecipes(ctx, "published", 25, &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 1 || page.NextCursor == nil || *page.NextCursor != "NEXT" {
		t.Fatalf("decoded page = %#v", page)
	}
	if err := client.ResearchRecipes(ctx, "published", 25, &page, "CURSOR"); err != nil {
		t.Fatal(err)
	}
	if err := client.ResearchRecipes(ctx, "published", 25, &page, ""); err != nil {
		t.Fatal(err)
	}

	if requests[0].URL.Query().Has("cursor") {
		t.Fatal("legacy recipe list request unexpectedly included a cursor")
	}
	if got := requests[1].URL.Query().Get("cursor"); got != "CURSOR" {
		t.Fatalf("cursor = %q, want CURSOR", got)
	}
	if !requests[2].URL.Query().Has("cursor") || requests[2].URL.Query().Get("cursor") != "" {
		t.Fatalf("explicit blank cursor was not preserved: %q", requests[2].URL.RawQuery)
	}
	if err := client.ResearchRecipes(ctx, "published", 25, &page, "a", "b"); err == nil {
		t.Fatal("multiple cursors must fail before making a request")
	}
	if len(requests) != 3 {
		t.Fatalf("invalid cursor variadics made requests: got %d, want 3", len(requests))
	}
}
