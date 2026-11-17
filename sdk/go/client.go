package baloot

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type Client struct {
	BaseURL, APIKey, TrustedKey         string
	SubjectID, SubjectName, AccessToken string
	HTTP                                *http.Client
}

type ResearchObserverSeat string

type ResearchBackendID string
type ResearchQualificationAuthority string
type RuntimeCompatibilityStatus string

const (
	ResearchObserverNorth ResearchObserverSeat = "north"
	ResearchObserverWest  ResearchObserverSeat = "west"
	ResearchObserverSouth ResearchObserverSeat = "south"
	ResearchObserverEast  ResearchObserverSeat = "east"

	ResearchBackendORToolsCPsatV2 ResearchBackendID              = "ortools_cp_sat_v2"
	ResearchQualificationCPsat    ResearchQualificationAuthority = "cp_sat"

	RuntimeCompatibilityNotChecked RuntimeCompatibilityStatus = "not_checked"
	RuntimeCompatibilityPassed     RuntimeCompatibilityStatus = "passed"
	RuntimeCompatibilityFailed     RuntimeCompatibilityStatus = "failed"
)

type ResearchRenderPrepareRequest struct {
	RevealPolicy string               `json:"revealPolicy,omitempty"`
	ExportPreset string               `json:"exportPreset,omitempty"`
	ObserverSeat ResearchObserverSeat `json:"observerSeat,omitempty"`
}

type ResearchRecipeListResponse struct {
	Items      []map[string]any `json:"items"`
	NextCursor *string          `json:"nextCursor"`
}

type ResearchAuthorityV1 struct {
	SchemaVersion          int                            `json:"schemaVersion"`
	BackendID              ResearchBackendID              `json:"backendId"`
	BackendVersion         string                         `json:"backendVersion"`
	QualificationAuthority ResearchQualificationAuthority `json:"qualificationAuthority"`
	SolverVersion          string                         `json:"solverVersion"`
	BundleRevisionID       string                         `json:"bundleRevisionId"`
	BundleHash             string                         `json:"bundleHash"`
	DomainIRHashes         map[string]string              `json:"domainIrHashes"`
	LoweringManifestHash   string                         `json:"loweringManifestHash"`
	ConstructionModelHash  string                         `json:"constructionModelHash"`
	ProofModelHash         string                         `json:"proofModelHash"`
	GraphHash              string                         `json:"graphHash"`
	TraceHash              string                         `json:"traceHash"`
	ProofOutcomeHash       string                         `json:"proofOutcomeHash"`
	SHA256                 string                         `json:"sha256"`
}

type RuntimeCompatibilityV1 struct {
	SchemaVersion int                        `json:"schemaVersion"`
	Status        RuntimeCompatibilityStatus `json:"status"`
	EngineVersion *string                    `json:"engineVersion"`
	BundleHash    string                     `json:"bundleHash"`
	TraceHash     string                     `json:"traceHash"`
	EvidenceHash  *string                    `json:"evidenceHash"`
	FailureCodes  []string                   `json:"failureCodes"`
	SHA256        string                     `json:"sha256"`
}

type ExecutionReceiptV1 struct {
	SchemaVersion      int                `json:"schemaVersion"`
	SemanticResultHash string             `json:"semanticResultHash"`
	StageTimings       map[string]float64 `json:"stageTimings"`
	CPUSeconds         float64            `json:"cpuSeconds"`
	PeakMemoryBytes    int64              `json:"peakMemoryBytes"`
	Conflicts          int64              `json:"conflicts"`
	Branches           int64              `json:"branches"`
	CacheBytes         int64              `json:"cacheBytes"`
	CompletedVariants  int64              `json:"completedVariants"`
	Machine            map[string]any     `json:"machine"`
	SHA256             string             `json:"sha256"`
}

type ResearchOperatorAuditV1Response struct {
	SchemaVersion        int                    `json:"schemaVersion"`
	ResultCode           string                 `json:"resultCode"`
	ResearchAuthority    ResearchAuthorityV1    `json:"researchAuthority"`
	RuntimeCompatibility RuntimeCompatibilityV1 `json:"runtimeCompatibility"`
	ExecutionReceipt     ExecutionReceiptV1     `json:"executionReceipt"`
}

type ResearchResultFilters struct {
	RunCode                    string
	TruthLabel                 string
	Disposition                string
	MinimumScore               *float64
	Search                     string
	TacticalTag                string
	ActingSeat                 string
	ObserverSeat               string
	Contract                   string
	Project                    string
	GenerationLane             string
	ScenarioTemplate           string
	ScenarioOrigin             string
	ScenarioStatus             string
	RequestedProofMode         string
	AchievedProofMode          string
	ProofCompletion            string
	ProofVisibility            string
	TargetTeam                 string
	ContractOutcome            string
	ProjectKind                string
	ProjectValue               string
	ProjectState               string
	WinningRootActionBand      string
	ExactStateAvailable        *bool
	DecisionKind               string
	ResearchBackend            string
	QualificationAuthority     string
	RuntimeCompatibilityStatus string
	ConfigurationBundle        string
	ExactStateID               string
	ScenarioCategory           string
	VariantCollection          string
	CreatedFrom                string
	CreatedTo                  string
	GroupBy                    string
	Sort                       string
	Offset                     *int
	Limit                      *int
}

func (filters ResearchResultFilters) Values() url.Values {
	values := url.Values{}
	for key, value := range map[string]string{
		"runCode": filters.RunCode, "truthLabel": filters.TruthLabel,
		"disposition": filters.Disposition, "search": filters.Search,
		"tacticalTag": filters.TacticalTag, "actingSeat": filters.ActingSeat,
		"observerSeat": filters.ObserverSeat, "contract": filters.Contract,
		"project": filters.Project, "generationLane": filters.GenerationLane,
		"scenarioTemplate": filters.ScenarioTemplate, "scenarioOrigin": filters.ScenarioOrigin,
		"scenarioStatus": filters.ScenarioStatus, "requestedProofMode": filters.RequestedProofMode,
		"achievedProofMode": filters.AchievedProofMode, "proofCompletion": filters.ProofCompletion,
		"proofVisibility": filters.ProofVisibility, "targetTeam": filters.TargetTeam,
		"contractOutcome": filters.ContractOutcome, "projectKind": filters.ProjectKind,
		"projectValue": filters.ProjectValue, "projectState": filters.ProjectState,
		"winningRootActionBand": filters.WinningRootActionBand, "decisionKind": filters.DecisionKind,
		"researchBackend": filters.ResearchBackend, "qualificationAuthority": filters.QualificationAuthority,
		"runtimeCompatibilityStatus": filters.RuntimeCompatibilityStatus, "configurationBundle": filters.ConfigurationBundle,
		"exactStateId": filters.ExactStateID, "scenarioCategory": filters.ScenarioCategory,
		"variantCollection": filters.VariantCollection, "createdFrom": filters.CreatedFrom,
		"createdTo": filters.CreatedTo, "groupBy": filters.GroupBy, "sort": filters.Sort,
	} {
		if value != "" {
			values.Set(key, value)
		}
	}
	if filters.MinimumScore != nil {
		values.Set("minimumScore", strconv.FormatFloat(*filters.MinimumScore, 'f', -1, 64))
	}
	if filters.ExactStateAvailable != nil {
		values.Set("exactStateAvailable", strconv.FormatBool(*filters.ExactStateAvailable))
	}
	if filters.Offset != nil {
		values.Set("offset", strconv.Itoa(*filters.Offset))
	}
	if filters.Limit != nil {
		values.Set("limit", strconv.Itoa(*filters.Limit))
	}
	return values
}

func New(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"), HTTP: http.DefaultClient}
}
func (c *Client) Get(ctx context.Context, path string, out any) error {
	return c.do(ctx, http.MethodGet, path, nil, out)
}
func (c *Client) Post(ctx context.Context, path string, payload, out any) error {
	return c.do(ctx, http.MethodPost, path, payload, out)
}
func (c *Client) Bundle(ctx context.Context, mode string, out any) error {
	return c.Get(ctx, "/api/v1/bundles/current?mode="+url.QueryEscape(mode), out)
}
func (c *Client) Health(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/health", out)
}
func (c *Client) Me(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/auth/me", out)
}
func (c *Client) Guest(ctx context.Context, displayName string, out any) error {
	return c.Post(ctx, "/api/v1/auth/guest", map[string]any{"displayName": displayName}, out)
}
func (c *Client) Register(ctx context.Context, username, password, displayName string, out any) error {
	return c.Post(ctx, "/api/v1/auth/register", map[string]any{"username": username, "password": password, "displayName": displayName}, out)
}
func (c *Client) Login(ctx context.Context, username, password string, out any) error {
	return c.Post(ctx, "/api/v1/auth/login", map[string]any{"username": username, "password": password}, out)
}
func (c *Client) Logout(ctx context.Context) error {
	return c.Post(ctx, "/api/v1/auth/logout", nil, nil)
}
func (c *Client) CreateGame(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/games", payload, out)
}
func (c *Client) Game(ctx context.Context, id string, reveal bool, out any) error {
	return c.Get(ctx, "/api/v1/games/"+url.PathEscape(id)+"?revealHands="+strconv.FormatBool(reveal), out)
}
func (c *Client) GetGame(ctx context.Context, id string, reveal bool, out any) error {
	return c.Game(ctx, id, reveal, out)
}
func (c *Client) Bid(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/games/"+url.PathEscape(id)+"/bids", payload, out)
}
func (c *Client) Play(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/games/"+url.PathEscape(id)+"/actions", payload, out)
}
func (c *Client) PlayCard(ctx context.Context, id string, payload, out any) error {
	return c.Play(ctx, id, payload, out)
}
func (c *Client) CreateLiveRoom(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/live/rooms", payload, out)
}
func (c *Client) WebSocketTicket(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/security/websocket-tickets", payload, out)
}
func (c *Client) Analyze(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/analyses", payload, out)
}
func (c *Client) CreateVideo(ctx context.Context, payload, out any, idempotencyKey ...string) error {
	key, err := mutationIdempotencyKey(idempotencyKey)
	if err != nil {
		return err
	}
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v1/video-jobs", payload, out, map[string]string{"Idempotency-Key": key})
}
func (c *Client) VideoJob(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/video-jobs/"+url.PathEscape(id), out)
}
func (c *Client) CancelVideo(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v1/video-jobs/"+url.PathEscape(id)+":cancel", nil, out)
}
func (c *Client) CreateFoundryAnalysisBatch(ctx context.Context, campaignHash string, out any) error {
	return c.Post(ctx, "/api/v2/analysis-batches", map[string]any{"campaignHash": campaignHash}, out)
}
func (c *Client) FoundryAnalysisBatch(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/analysis-batches/"+url.PathEscape(id), out)
}
func (c *Client) CancelFoundryAnalysisBatch(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v2/analysis-batches/"+url.PathEscape(id)+":cancel", nil, out)
}
func (c *Client) CreateRenderBatch(ctx context.Context, items any, idempotencyKey string, out any) error {
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/render-batches", map[string]any{"items": items}, out, map[string]string{"Idempotency-Key": idempotencyKey})
}
func (c *Client) RenderBatch(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/render-batches/"+url.PathEscape(id), out)
}
func (c *Client) CancelRenderBatch(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v2/render-batches/"+url.PathEscape(id)+":cancel", nil, out)
}
func (c *Client) CreateContextSearch(ctx context.Context, parentTraceReference any, decisionID string, contextProofHashes []string, idempotencyKey string, out any) error {
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/context-searches", map[string]any{"parentTraceReference": parentTraceReference, "decisionId": decisionID, "contextProofHashes": contextProofHashes}, out, map[string]string{"Idempotency-Key": idempotencyKey})
}
func (c *Client) ContextSearch(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/context-searches/"+url.PathEscape(id), out)
}
func (c *Client) CancelContextSearch(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v2/context-searches/"+url.PathEscape(id)+":cancel", nil, out)
}
func (c *Client) CreateNaturalSakkaGeneration(ctx context.Context, rootSeed int64, initialDealer, idempotencyKey string, out any) error {
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/sakka-generations", map[string]any{"rootSeed": rootSeed, "initialDealer": initialDealer}, out, map[string]string{"Idempotency-Key": idempotencyKey})
}
func (c *Client) NaturalSakkaGeneration(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/sakka-generations/"+url.PathEscape(id), out)
}
func (c *Client) CancelNaturalSakkaGeneration(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v2/sakka-generations/"+url.PathEscape(id)+":cancel", nil, out)
}
func (c *Client) FoundrySakkas(ctx context.Context, provenance, after string, limit int, out any) error {
	query := url.Values{"limit": []string{strconv.Itoa(limit)}}
	if provenance != "" {
		query.Set("provenance", provenance)
	}
	if after != "" {
		query.Set("after", after)
	}
	return c.Get(ctx, "/api/v2/sakkas?"+query.Encode(), out)
}
func (c *Client) CreateNaturalSakka(ctx context.Context, traceReferences any, target int, out any) error {
	return c.Post(ctx, "/api/v2/sakkas/natural", map[string]any{"traceReferences": traceReferences, "target": target}, out)
}
func (c *Client) CreateCuratedSakka(ctx context.Context, selections any, target int, out any) error {
	return c.Post(ctx, "/api/v2/sakkas/curated", map[string]any{"selections": selections, "target": target}, out)
}
func (c *Client) FoundrySakka(ctx context.Context, sakkaHash string, out any) error {
	return c.Get(ctx, "/api/v2/sakkas/"+url.PathEscape(sakkaHash), out)
}
func (c *Client) CreateCampaignEquivalenceProof(ctx context.Context, executions, out any) error {
	return c.Post(ctx, "/api/v2/campaign-equivalence-proofs", map[string]any{"executions": executions}, out)
}
func (c *Client) FoundryCampaignAttempts(ctx context.Context, campaignHash string, out any) error {
	return c.Get(ctx, "/api/v2/campaigns/"+url.PathEscape(campaignHash)+"/attempts", out)
}
func (c *Client) FoundryProfile(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v2/foundry-profile", out)
}
func (c *Client) FoundryCampaigns(ctx context.Context, after string, limit int, out any) error {
	query := url.Values{"limit": []string{strconv.Itoa(limit)}}
	if after != "" {
		query.Set("after", after)
	}
	return c.Get(ctx, "/api/v2/campaigns?"+query.Encode(), out)
}
func (c *Client) CreateTraceImportCatalog(ctx context.Context, lane string, entries, out any) error {
	return c.Post(ctx, "/api/v2/trace-import-catalogs", map[string]any{"lane": lane, "entries": entries}, out)
}
func (c *Client) CreateScoreReachabilityCatalog(ctx context.Context, traceReferences, out any) error {
	return c.Post(ctx, "/api/v2/score-reachability-catalogs", map[string]any{"traceReferences": traceReferences}, out)
}
func (c *Client) ProveScoreReachability(ctx context.Context, catalogHash string, targetScore [2]int, targetDealer, initialDealer string, out any) error {
	return c.Post(ctx, "/api/v2/score-reachability-catalogs/"+url.PathEscape(catalogHash)+":prove", map[string]any{"targetScore": targetScore, "targetDealer": targetDealer, "initialDealer": initialDealer}, out)
}
func (c *Client) FoundryCatalogs(ctx context.Context, kind, after string, limit int, out any) error {
	query := url.Values{"limit": []string{strconv.Itoa(limit)}}
	if kind != "" {
		query.Set("kind", kind)
	}
	if after != "" {
		query.Set("after", after)
	}
	return c.Get(ctx, "/api/v2/foundry-catalogs?"+query.Encode(), out)
}
func (c *Client) FoundryCatalog(ctx context.Context, catalogHash string, out any) error {
	return c.Get(ctx, "/api/v2/foundry-catalogs/"+url.PathEscape(catalogHash), out)
}
func (c *Client) CreateChallengeLocator(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v2/challenge-locators", payload, out)
}
func (c *Client) ChallengeLocator(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/challenge-locators/"+url.PathEscape(code), out)
}
func (c *Client) CommitChallengeLocatorAttempt(ctx context.Context, code string, payload, out any) error {
	return c.Post(ctx, "/api/v2/challenge-locators/"+url.PathEscape(code)+"/attempts", payload, out)
}
func (c *Client) RevokeChallengeLocator(ctx context.Context, code string, out any) error {
	return c.Post(ctx, "/api/v2/challenge-locators/"+url.PathEscape(code)+":revoke", nil, out)
}

// Unified domain compiler and Research Foundry V2.
func (c *Client) DomainCapabilities(ctx context.Context, includePayloads bool, out any, bundleRevisionID ...string) error {
	if len(bundleRevisionID) > 1 {
		return fmt.Errorf("baloot: domain capabilities accepts at most one bundle revision ID")
	}
	query := url.Values{"includePayloads": []string{strconv.FormatBool(includePayloads)}}
	if len(bundleRevisionID) == 1 {
		query.Set("bundleRevisionId", bundleRevisionID[0])
	}
	return c.Get(ctx, "/api/v2/domain-capabilities?"+query.Encode(), out)
}
func (c *Client) DomainCompilation(ctx context.Context, hash string, includePayloads bool, out any) error {
	return c.Get(ctx, "/api/v2/domain-compilations/"+url.PathEscape(hash)+"?includePayloads="+strconv.FormatBool(includePayloads), out)
}
func (c *Client) DomainCompilationCoverage(ctx context.Context, hash string, out any) error {
	return c.Get(ctx, "/api/v2/domain-compilations/"+url.PathEscape(hash)+"/coverage", out)
}
func (c *Client) RulePrimitives(ctx context.Context, out any, bundleRevisionID ...string) error {
	if len(bundleRevisionID) > 1 {
		return fmt.Errorf("baloot: rule primitives accepts at most one bundle revision ID")
	}
	if len(bundleRevisionID) == 0 {
		return c.Get(ctx, "/api/v2/rule-primitives", out)
	}
	query := url.Values{"bundleRevisionId": []string{bundleRevisionID[0]}}
	return c.Get(ctx, "/api/v2/rule-primitives?"+query.Encode(), out)
}
func (c *Client) ValidateResearchRecipe(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v2/research-recipes:validate", payload, out)
}
func (c *Client) EstimateResearchRecipe(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v2/research-recipes:estimate", payload, out)
}
func (c *Client) CreateResearchRecipe(ctx context.Context, payload, out any, idempotencyKey ...string) error {
	key, err := mutationIdempotencyKey(idempotencyKey)
	if err != nil {
		return err
	}
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/research-recipes", payload, out, map[string]string{"Idempotency-Key": key})
}
func (c *Client) ResearchRecipes(ctx context.Context, status string, limit int, out any, cursor ...string) error {
	if len(cursor) > 1 {
		return fmt.Errorf("baloot: research recipes accepts at most one cursor")
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}}
	if status != "" {
		q.Set("status", status)
	}
	if len(cursor) == 1 {
		q.Set("cursor", cursor[0])
	}
	return c.Get(ctx, "/api/v2/research-recipes?"+q.Encode(), out)
}
func (c *Client) ResearchRecipe(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-recipes/"+url.PathEscape(code), out)
}
func (c *Client) RetireResearchRecipe(ctx context.Context, code string, out any, idempotencyKey ...string) error {
	key, err := mutationIdempotencyKey(idempotencyKey)
	if err != nil {
		return err
	}
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/research-recipes/"+url.PathEscape(code)+":retire", nil, out, map[string]string{"Idempotency-Key": key})
}
func (c *Client) CreateResearchRun(ctx context.Context, recipeCode, key string, out any) error {
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/research-runs", map[string]any{"recipeCode": recipeCode}, out, map[string]string{"Idempotency-Key": key})
}
func (c *Client) ResearchRuns(ctx context.Context, status string, limit int, out any) error {
	q := url.Values{"limit": {strconv.Itoa(limit)}}
	if status != "" {
		q.Set("status", status)
	}
	return c.Get(ctx, "/api/v2/research-runs?"+q.Encode(), out)
}
func (c *Client) ResearchRun(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-runs/"+url.PathEscape(code), out)
}
func (c *Client) ResearchRunStats(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-runs/"+url.PathEscape(code)+"/stats", out)
}
func (c *Client) ResearchStats(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v2/research-stats", out)
}
func (c *Client) CancelResearchRun(ctx context.Context, code string, out any) error {
	return c.Post(ctx, "/api/v2/research-runs/"+url.PathEscape(code)+":cancel", nil, out)
}
func (c *Client) ResearchRunEvents(ctx context.Context, code string, handle func(map[string]any) error) error {
	return c.streamEvents(ctx, "/api/v2/research-runs/"+url.PathEscape(code)+"/events", handle)
}
func (c *Client) CreateComparisonCampaign(ctx context.Context, recipeCode, key string, out any) error {
	return c.doWithHeaders(ctx, http.MethodPost, "/api/v2/comparison-campaigns", map[string]any{"recipeCode": recipeCode}, out, map[string]string{"Idempotency-Key": key})
}
func (c *Client) ComparisonCampaigns(ctx context.Context, limit int, out any) error {
	return c.Get(ctx, "/api/v2/comparison-campaigns?limit="+strconv.Itoa(limit), out)
}
func (c *Client) ComparisonCampaign(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/comparison-campaigns/"+url.PathEscape(code), out)
}
func (c *Client) ComparisonCampaignEvents(ctx context.Context, code string, handle func(map[string]any) error) error {
	return c.streamEvents(ctx, "/api/v2/comparison-campaigns/"+url.PathEscape(code)+"/events", handle)
}
func (c *Client) ResearchResults(ctx context.Context, q url.Values, out any) error {
	if q == nil {
		q = url.Values{}
	}
	if !q.Has("sort") {
		q.Set("sort", "score_desc")
	}
	if !q.Has("offset") {
		q.Set("offset", "0")
	}
	if !q.Has("limit") {
		q.Set("limit", "100")
	}
	return c.Get(ctx, "/api/v2/research-results?"+q.Encode(), out)
}
func (c *Client) ResearchResultsWithFilters(ctx context.Context, filters ResearchResultFilters, out any) error {
	return c.ResearchResults(ctx, filters.Values(), out)
}
func (c *Client) ResearchResult(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code), out)
}
func (c *Client) ResearchResultOperatorAudit(ctx context.Context, code string, out *ResearchOperatorAuditV1Response) error {
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/operator-audit", out)
}
func (c *Client) ResearchResultDecisions(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/decisions", out)
}
func (c *Client) ResearchResultPerspectives(ctx context.Context, code, observerSeat string, out any) error {
	q := url.Values{"observerSeat": []string{observerSeat}}
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/perspectives?"+q.Encode(), out)
}
func (c *Client) ResearchResultDecision(ctx context.Context, code string, ordinal int, out any) error {
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/decisions/"+strconv.Itoa(ordinal), out)
}
func (c *Client) PrepareResearchResultRender(ctx context.Context, code string, payload, out any) error {
	if payload == nil {
		payload = map[string]any{}
	}
	return c.Post(ctx, "/api/v2/research-results/"+url.PathEscape(code)+":prepare-render", payload, out)
}
func (c *Client) PrepareResearchDecisionRender(ctx context.Context, code string, ordinal int, payload, out any) error {
	if payload == nil {
		payload = map[string]any{}
	}
	return c.Post(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/decisions/"+strconv.Itoa(ordinal)+":prepare-render", payload, out)
}
func (c *Client) ResearchResultRenders(ctx context.Context, code string, out any) error {
	return c.Get(ctx, "/api/v2/research-results/"+url.PathEscape(code)+"/renders", out)
}
func (c *Client) Laffat(ctx context.Context, q url.Values, out any) error {
	if q == nil {
		q = url.Values{}
	}
	if !q.Has("sort") {
		q.Set("sort", "newest")
	}
	if !q.Has("offset") {
		q.Set("offset", "0")
	}
	if !q.Has("limit") {
		q.Set("limit", "100")
	}
	return c.Get(ctx, "/api/v2/laffat?"+q.Encode(), out)
}
func (c *Client) Laffa(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/laffat/"+url.PathEscape(id), out)
}
func (c *Client) PrepareLaffaRender(ctx context.Context, id string, payload, out any) error {
	if payload == nil {
		payload = map[string]any{}
	}
	return c.Post(ctx, "/api/v2/laffat/"+url.PathEscape(id)+":prepare-render", payload, out)
}
func (c *Client) LaffaRenders(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v2/laffat/"+url.PathEscape(id)+"/renders", out)
}
func (c *Client) ArchivedGame(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/history/games/"+url.PathEscape(id), out)
}
func (c *Client) ReplayPackage(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/history/games/"+url.PathEscape(id)+"/replay", out)
}
func (c *Client) AuditGame(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/history/games/"+url.PathEscape(id)+":audit", payload, out)
}
func (c *Client) ExplainDecision(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/history/decisions/"+url.PathEscape(id)+":explain", payload, out)
}
func (c *Client) CompareDecision(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/history/decisions/"+url.PathEscape(id)+"/comparison", out)
}
func (c *Client) AnnotateIntent(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/history/decisions/"+url.PathEscape(id)+"/intent", payload, out)
}
func (c *Client) RecordDecisionAttempt(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/history/decisions/"+url.PathEscape(id)+"/attempts", payload, out)
}
func (c *Client) DecisionDifficulty(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/history/decisions/"+url.PathEscape(id)+"/difficulty", out)
}
func (c *Client) RecordChallengeAttempt(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/challenges/"+url.PathEscape(id)+"/attempts", payload, out)
}
func (c *Client) ChallengeDifficulty(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/challenges/"+url.PathEscape(id)+"/difficulty", out)
}
func (c *Client) AdaptiveChallenge(ctx context.Context, playerID string, recent []string, out any) error {
	q := url.Values{}
	if playerID != "" {
		q.Set("playerId", playerID)
	}
	q.Set("recent", strings.Join(recent, ","))
	return c.Get(ctx, "/api/v1/adaptive/challenges/next?"+q.Encode(), out)
}
func (c *Client) VideoRecord(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/videos/"+url.PathEscape(id), out)
}
func (c *Client) VideoTimeline(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/videos/"+url.PathEscape(id)+"/timeline", out)
}
func (c *Client) VideoAuditMap(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/videos/"+url.PathEscape(id)+"/audit-map", out)
}

// Canonical standardized shared play: Laffa (one round) or Sakka (full match).
func (c *Client) SharedBoards(ctx context.Context, limit int, out any) error {
	return c.Get(ctx, "/api/v1/shared/boards?limit="+strconv.Itoa(limit), out)
}
func (c *Client) CreateSharedBoard(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/shared/boards", payload, out)
}
func (c *Client) SharedBoard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/boards/"+url.PathEscape(id), out)
}
func (c *Client) VerifySharedBoard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/boards/"+url.PathEscape(id)+"/verification", out)
}
func (c *Client) CreateSharedRoom(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/shared/rooms", payload, out)
}
func (c *Client) SharedRoom(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/rooms/"+url.PathEscape(id), out)
}
func (c *Client) JoinSharedRoom(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v1/shared/rooms/"+url.PathEscape(id)+":join", map[string]any{}, out)
}
func (c *Client) StartSharedRun(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v1/shared/rooms/"+url.PathEscape(id)+"/runs", map[string]any{}, out)
}
func (c *Client) SharedRun(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/runs/"+url.PathEscape(id), out)
}
func (c *Client) VerifySharedRun(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/runs/"+url.PathEscape(id)+"/verification", out)
}
func (c *Client) PlaySharedAction(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/shared/runs/"+url.PathEscape(id)+"/actions", payload, out)
}
func (c *Client) SharedRunReview(ctx context.Context, id string, ensureAudit bool, out any) error {
	return c.Get(ctx, "/api/v1/shared/runs/"+url.PathEscape(id)+"/review?ensureAudit="+strconv.FormatBool(ensureAudit), out)
}
func (c *Client) ExplainSharedDecision(ctx context.Context, runID, decisionID string, options url.Values, out any) error {
	if options == nil {
		options = url.Values{}
	}
	if !options.Has("level") {
		options.Set("level", "basic")
	}
	if !options.Has("surface") {
		options.Set("surface", "replay")
	}
	if !options.Has("viewMode") {
		options.Set("viewMode", "observer")
	}
	return c.Get(ctx, "/api/v1/shared/runs/"+url.PathEscape(runID)+"/decisions/"+url.PathEscape(decisionID)+"/explanation?"+options.Encode(), out)
}
func (c *Client) SharedDecisionReviews(ctx context.Context, runID, decisionID string, out any) error {
	return c.Get(ctx, "/api/v1/shared/runs/"+url.PathEscape(runID)+"/decisions/"+url.PathEscape(decisionID)+"/reviews", out)
}
func (c *Client) ReviewSharedDecision(ctx context.Context, runID, decisionID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/shared/runs/"+url.PathEscape(runID)+"/decisions/"+url.PathEscape(decisionID)+"/reviews", payload, out)
}
func (c *Client) ShareSharedDecision(ctx context.Context, runID string, sequence int, payload, out any) error {
	return c.Post(ctx, "/api/v1/shared/runs/"+url.PathEscape(runID)+"/actions/"+strconv.Itoa(sequence)+":share", payload, out)
}
func (c *Client) SharedLeaderboard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/shared/rooms/"+url.PathEscape(id)+"/leaderboard", out)
}
func (c *Client) CompareSharedRuns(ctx context.Context, runA, runB string, out any) error {
	return c.Post(ctx, "/api/v1/shared/comparisons", map[string]any{"runA": runA, "runB": runB}, out)
}

func (c *Client) SakkaBoards(ctx context.Context, limit int, out any) error {
	return c.Get(ctx, "/api/v1/sakka/boards?limit="+strconv.Itoa(limit), out)
}
func (c *Client) CreateSakkaBoard(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/boards", payload, out)
}
func (c *Client) SakkaBoard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/boards/"+url.PathEscape(id), out)
}
func (c *Client) VerifySakkaBoard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/boards/"+url.PathEscape(id)+"/verification", out)
}
func (c *Client) CreateSakkaRoom(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/rooms", payload, out)
}
func (c *Client) SakkaRoom(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/rooms/"+url.PathEscape(id), out)
}
func (c *Client) JoinSakkaRoom(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v1/sakka/rooms/"+url.PathEscape(id)+":join", map[string]any{}, out)
}
func (c *Client) StartSakkaRun(ctx context.Context, id string, out any) error {
	return c.Post(ctx, "/api/v1/sakka/rooms/"+url.PathEscape(id)+"/runs", map[string]any{}, out)
}
func (c *Client) SakkaRun(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id), out)
}
func (c *Client) VerifySakkaRun(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id)+"/verification", out)
}
func (c *Client) PlaySakkaAction(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id)+"/actions", payload, out)
}
func (c *Client) AuditSakkaRun(ctx context.Context, id string, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id)+":audit", payload, out)
}
func (c *Client) ShareSakkaDecision(ctx context.Context, id string, sequence int, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id)+"/actions/"+strconv.Itoa(sequence)+":share", payload, out)
}
func (c *Client) SakkaFeedback(ctx context.Context, id string, sequence int, payload, out any) error {
	return c.Post(ctx, "/api/v1/sakka/runs/"+url.PathEscape(id)+"/actions/"+strconv.Itoa(sequence)+"/feedback", payload, out)
}
func (c *Client) SakkaLeaderboard(ctx context.Context, id string, out any) error {
	return c.Get(ctx, "/api/v1/sakka/rooms/"+url.PathEscape(id)+"/leaderboard", out)
}

func (c *Client) CompareSakkaRuns(ctx context.Context, runA, runB string, out any) error {
	return c.Post(ctx, "/api/v1/sakka/comparisons", map[string]any{"runA": runA, "runB": runB}, out)
}

// Governed Human-in-the-Loop V18.
func (c *Client) HitlProfile(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/hitl/profile", out)
}
func (c *Client) HitlCases(ctx context.Context, limit int, out any) error {
	return c.Get(ctx, "/api/v1/hitl/cases?limit="+strconv.Itoa(limit), out)
}
func (c *Client) CreateHitlCase(ctx context.Context, decisionID string, highImpact bool, out any) error {
	return c.CreateTypedHitlCase(ctx, decisionID, "analysis_defect", highImpact, out)
}
func (c *Client) CreateTypedHitlCase(ctx context.Context, decisionID, correctionType string, highImpact bool, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases", map[string]any{"decisionId": decisionID, "correctionType": correctionType, "highImpact": highImpact}, out)
}
func (c *Client) HitlCase(ctx context.Context, caseID string, blind bool, out any) error {
	return c.Get(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"?blind="+strconv.FormatBool(blind), out)
}
func (c *Client) AssignHitlReview(ctx context.Context, caseID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"/assignments", payload, out)
}
func (c *Client) SubmitHitlReview(ctx context.Context, caseID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"/submissions", payload, out)
}
func (c *Client) ComputeHitlConsensus(ctx context.Context, caseID string, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"/consensus", map[string]any{}, out)
}
func (c *Client) CreateHitlTrainingCandidate(ctx context.Context, caseID string, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"/training-candidates", map[string]any{}, out)
}
func (c *Client) EvaluateHitlCandidate(ctx context.Context, candidateID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/training-candidates/"+url.PathEscape(candidateID)+"/evaluations", payload, out)
}
func (c *Client) RenderHitlReviewClip(ctx context.Context, caseID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/cases/"+url.PathEscape(caseID)+"/review-clip", payload, out)
}
func (c *Client) HitlImprovements(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/hitl/improvements", out)
}
func (c *Client) CreateHitlImprovement(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/improvements", payload, out)
}
func (c *Client) EvaluateHitlImprovement(ctx context.Context, proposalID string, payload, out any) error {
	return c.Post(ctx, "/api/v1/hitl/improvements/"+url.PathEscape(proposalID)+"/evaluations", payload, out)
}

// Versioned row-based configuration control plane V19.
func (c *Client) ConfigurationCapabilities(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/configuration-registry/profile", out)
}
func (c *Client) ConfigurationProfiles(ctx context.Context, values url.Values, out any) error {
	if values == nil {
		values = url.Values{}
	}
	if !values.Has("limit") {
		values.Set("limit", "200")
	}
	return c.Get(ctx, "/api/v1/configuration-registry/profiles?"+values.Encode(), out)
}
func (c *Client) CreateConfigurationProfile(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/profiles", payload, out)
}
func (c *Client) PublishConfigurationProfile(ctx context.Context, revisionID string, expectedEtag int64, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/profiles/"+url.PathEscape(revisionID)+":publish", map[string]any{"expectedEtag": expectedEtag}, out)
}
func (c *Client) RejectConfigurationProfile(ctx context.Context, revisionID string, expectedEtag int64, reason string, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/profiles/"+url.PathEscape(revisionID)+":reject", map[string]any{"expectedEtag": expectedEtag, "reason": reason}, out)
}
func (c *Client) RetireConfigurationProfile(ctx context.Context, revisionID string, expectedEtag int64, reason string, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/profiles/"+url.PathEscape(revisionID)+":retire", map[string]any{"expectedEtag": expectedEtag, "reason": reason}, out)
}
func (c *Client) ResolveConfigurationProfile(ctx context.Context, family, profileID, tenantID string, out any) error {
	q := url.Values{"family": {family}, "profileId": {profileID}}
	if tenantID != "" {
		q.Set("tenantId", tenantID)
	}
	return c.Get(ctx, "/api/v1/configuration-registry/profiles:resolve?"+q.Encode(), out)
}
func (c *Client) CompareConfigurationProfiles(ctx context.Context, left, right string, out any) error {
	q := url.Values{"left": {left}, "right": {right}}
	return c.Get(ctx, "/api/v1/configuration-registry/profiles:compare?"+q.Encode(), out)
}
func (c *Client) ConfigurationBundles(ctx context.Context, values url.Values, out any) error {
	if values == nil {
		values = url.Values{}
	}
	if !values.Has("limit") {
		values.Set("limit", "200")
	}
	return c.Get(ctx, "/api/v1/configuration-registry/bundles?"+values.Encode(), out)
}
func (c *Client) CreateConfigurationBundle(ctx context.Context, payload, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/bundles", payload, out)
}
func (c *Client) PublishConfigurationBundle(ctx context.Context, revisionID string, expectedEtag int64, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/bundles/"+url.PathEscape(revisionID)+":publish", map[string]any{"expectedEtag": expectedEtag}, out)
}
func (c *Client) PublishConfiguration(ctx context.Context, revisionID string, expectedEtag int64, out any) error {
	return c.PublishConfigurationBundle(ctx, revisionID, expectedEtag, out)
}
func (c *Client) RejectConfigurationBundle(ctx context.Context, revisionID string, expectedEtag int64, reason string, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/bundles/"+url.PathEscape(revisionID)+":reject", map[string]any{"expectedEtag": expectedEtag, "reason": reason}, out)
}
func (c *Client) RetireConfigurationBundle(ctx context.Context, revisionID string, expectedEtag int64, reason string, out any) error {
	return c.Post(ctx, "/api/v1/configuration-registry/bundles/"+url.PathEscape(revisionID)+":retire", map[string]any{"expectedEtag": expectedEtag, "reason": reason}, out)
}
func (c *Client) ResolveConfigurationBundle(ctx context.Context, bundleID, tenantID string, out any) error {
	q := url.Values{}
	if bundleID != "" {
		q.Set("bundleId", bundleID)
	}
	if tenantID != "" {
		q.Set("tenantId", tenantID)
	}
	return c.Get(ctx, "/api/v1/configuration-registry/resolve?"+q.Encode(), out)
}
func (c *Client) ResolveConfiguration(ctx context.Context, bundleID, tenantID string, out any) error {
	return c.ResolveConfigurationBundle(ctx, bundleID, tenantID, out)
}
func (c *Client) ResolveConfigurationBundleRevision(ctx context.Context, revisionID string, allowDraft bool, out any) error {
	q := url.Values{"allowDraft": {strconv.FormatBool(allowDraft)}}
	return c.Get(ctx, "/api/v1/configuration-registry/bundles/"+url.PathEscape(revisionID)+":resolve?"+q.Encode(), out)
}
func (c *Client) Plugins(ctx context.Context, kind string, out any) error {
	path := "/api/v1/plugins"
	if kind != "" {
		path += "?kind=" + url.QueryEscape(kind)
	}
	return c.Get(ctx, path, out)
}
func (c *Client) ListPlugins(ctx context.Context, out any) error {
	return c.Plugins(ctx, "", out)
}
func (c *Client) PlatformCapabilities(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/platform/capabilities", out)
}
func (c *Client) PerformanceProfile(ctx context.Context, out any) error {
	return c.Get(ctx, "/api/v1/performance/profile", out)
}
func (c *Client) RunPerformanceQuick(ctx context.Context, out any) error {
	return c.Post(ctx, "/api/v1/performance/quick", map[string]any{}, out)
}

func (c *Client) do(ctx context.Context, method, path string, payload, out any) error {
	return c.doWithHeaders(ctx, method, path, payload, out, nil)
}

func mutationIdempotencyKey(values []string) (string, error) {
	if len(values) > 1 {
		return "", fmt.Errorf("baloot: mutation accepts at most one idempotency key")
	}
	if len(values) == 1 {
		return values[0], nil
	}
	token := make([]byte, 16)
	if _, err := rand.Read(token); err != nil {
		return "", fmt.Errorf("baloot: generate mutation idempotency key: %w", err)
	}
	return hex.EncodeToString(token), nil
}

func (c *Client) doWithHeaders(ctx context.Context, method, path string, payload, out any, customHeaders map[string]string) error {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, body)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
	if c.TrustedKey != "" {
		req.Header.Set("X-Baloot-Trusted-Key", c.TrustedKey)
	}
	if c.SubjectID != "" {
		req.Header.Set("X-Baloot-Subject-ID", c.SubjectID)
	}
	if c.SubjectName != "" {
		req.Header.Set("X-Baloot-Subject-Name", c.SubjectName)
	}
	if c.AccessToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AccessToken)
	}
	for key, value := range customHeaders {
		req.Header.Set(key, value)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("baloot: %s: %s", resp.Status, string(data))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *Client) streamEvents(ctx context.Context, path string, handle func(map[string]any) error) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")
	if c.APIKey != "" {
		req.Header.Set("X-API-Key", c.APIKey)
	}
	if c.TrustedKey != "" {
		req.Header.Set("X-Baloot-Trusted-Key", c.TrustedKey)
	}
	if c.SubjectID != "" {
		req.Header.Set("X-Baloot-Subject-ID", c.SubjectID)
	}
	if c.SubjectName != "" {
		req.Header.Set("X-Baloot-Subject-Name", c.SubjectName)
	}
	if c.AccessToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AccessToken)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("baloot: %s: %s", resp.Status, string(data))
	}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			return err
		}
		if handle != nil {
			if err := handle(event); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}
