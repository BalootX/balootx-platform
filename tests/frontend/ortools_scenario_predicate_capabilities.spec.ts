import {describe, expect, test} from 'vitest';

import type {
  ScenarioPredicateCapabilityV3,
  ScenarioPredicateDefinitionV2,
  ScenarioResearchV4CapabilityV1,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model.js';
import {
  mergeScenarioPredicateDefinitions,
  scenarioPredicateArgumentValues,
  scenarioPredicateCausalVerificationAvailable,
  scenarioPredicateSupported,
  unsupportedScenarioPredicateIds,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/ortoolsScenarioUi.js';

const hash = 'a'.repeat(64);

function definition(kind: string): ScenarioPredicateDefinitionV2 {
  return {
    schemaVersion: 2,
    kind,
    version: 1,
    category: 'proof_tactic',
    stage: 'proof',
    argumentSchema: {type: 'object', additionalProperties: false},
    messageArKey: 'predicate.ar',
    messageEnKey: 'predicate.en',
    requiredDomainPrimitives: [],
    allowedHardness: ['hard'],
    cpSatSupport: {
      schemaVersion: 1,
      search: {status: 'exact', stages: ['proof']},
      exactInput: {status: 'exact', stages: ['proof']},
      qualificationSource: 'proof',
      actorInformationSet: 'public',
      proof: {exact: 'exact', bounded: 'bounds', policyConditioned: 'exact'},
      evidenceExtractor: 'evidence.v1',
    },
    costClass: 'solver',
    safeComplement: false,
    examples: [],
    counterexamples: [],
    documentation: [],
    tests: [],
    lifecycleStatus: 'active',
    sha256: hash,
  };
}

function predicateCapability(
  kind: string,
  decisionKind: 'bid' | 'play',
  lineUniverses: ScenarioPredicateCapabilityV3['executionContexts'][number]['lineUniverses'],
): ScenarioPredicateCapabilityV3 {
  return {
    schemaVersion: 3,
    kind,
    definitionSha256: hash,
    registrationSha256: hash,
    lowererOwner: 'lowerer.v1',
    evidenceExtractor: 'evidence.v1',
    owningTest: 'test_predicate',
    qualificationBinding: 'proof_variable',
    requiredRulePrimitives: [],
    nativeStatus: 'native_exact',
    origins: ['search'],
    decisionKinds: [decisionKind],
    proofModes: ['exact'],
    visibilityClasses: ['complete_state'],
    executionContexts: [{
      origin: 'search',
      decisionKind,
      proofMode: 'exact',
      visibility: 'complete_state',
      lineUniverses,
    }],
    sha256: hash,
  };
}

function capability(
  capabilities: ScenarioPredicateCapabilityV3[] | undefined,
): ScenarioResearchV4CapabilityV1 {
  const definitions = [
    definition('adjudication.purchase_before_turn_attempt'),
    definition('proof.winning_continuation_line_count'),
    definition('tactic.applicable'),
  ];
  return {
    schemaVersion: 1,
    backendIdentity: {
      schemaVersion: 1,
      backendId: 'ortools_cp_sat_v3',
      backendVersion: '3.0.0',
      qualificationAuthority: 'cp_sat',
      solverVersion: '9.15',
      planner: 'tail_first_v1',
      deterministicExecution: true,
      bundleRevisionId: 'CBR-TEST',
      bundleIdentityHash: hash,
      authorityHash: hash,
      predicateCatalogHash: hash,
      loweringManifestHash: hash,
      sha256: hash,
    },
    executionReady: true,
    launchAdmissible: true,
    supportedOrigins: ['search'],
    supportedProofVisibility: [{mode: 'exact', visibility: 'complete_state'}],
    executionCapabilities: {
      schemaVersion: 1,
      combinations: [],
      decisionSelectorSchemas: {},
      authoringCatalog: {
        ...(capabilities ? {
          predicateCapabilityManifest: {
            schemaVersion: 3,
            registryVersion: 'scenario_predicate_capability_registry_v3',
            backendId: 'ortools_cp_sat_v3',
            baseManifestHash: hash,
            capabilities,
            sha256: hash,
          },
        } : {}),
      } as ScenarioResearchV4CapabilityV1['executionCapabilities']['authoringCatalog'],
      sha256: hash,
    },
    hardPredicateCatalog: {
      schemaVersion: 2,
      backendIdentityHash: hash,
      loweringManifestHash: hash,
      origin: 'search',
      proofMode: 'exact',
      visibility: 'complete_state',
      hardness: 'hard',
      definitions,
      sha256: hash,
    },
    localization: {
      schemaVersion: 1,
      bundleRevisionId: 'CBR-TEST',
      productRevisionId: 'CPR-TEST',
      productProfileHash: hash,
      locale: 'ar-SA',
      direction: 'rtl',
      groups: {workflow: {}},
      sha256: hash,
    },
    sha256: hash,
  };
}

describe('V3 predicate execution contexts', () => {
  test('rejects unavailable hard causality without rewriting imported arguments', () => {
    const tactic = predicateCapability('tactic.verified', 'play', []);
    tactic.argumentContexts = ['large_card_sacrifice', 'small_card_big_gain'].map((argumentValue) => ({
      argumentName: 'kind', argumentValue, decisionKinds: ['play'],
      requiredUtility: null,
      requiredProofBindings: [{binding: 'principal_variation_has_continuation_variable', minimumCount: 1}],
      causalVerificationAvailable: false,
    }));
    const source = capability([tactic]);
    source.hardPredicateCatalog.definitions.push({
      ...definition('tactic.verified'), allowedHardness: ['hard', 'prefer', 'report'],
    });
    const merged = mergeScenarioPredicateDefinitions([], source);
    const row = merged.find((item) => item.kind === 'tactic.verified')!;
    const selection = {
      origin: 'search', decisionKind: 'play', proofMode: 'exact', visibility: 'complete_state',
    } as const;
    for (const kind of ['large_card_sacrifice', 'small_card_big_gain']) {
      const args = Object.freeze({kind, causalRequired: true});
      expect(scenarioPredicateCausalVerificationAvailable(row, args)).toBe(false);
      expect(scenarioPredicateSupported(row, selection, 'hard', args)).toBe(false);
      expect(unsupportedScenarioPredicateIds(merged, [{
        predicateId: 'PRED-CAUSAL', kind: row.kind, hardness: 'hard', arguments: args,
      }], selection)).toEqual(['PRED-CAUSAL']);
      expect(args.causalRequired).toBe(true);
      expect(scenarioPredicateSupported(row, selection, 'hard', {...args, causalRequired: false})).toBe(true);
      expect(scenarioPredicateSupported(row, selection, 'hard', {kind})).toBe(true);
      expect(scenarioPredicateSupported(row, selection, 'report', args)).toBe(true);
      expect(scenarioPredicateSupported(row, selection, 'prefer', args)).toBe(true);
    }
    const historical = {...row, v3ExecutionCapability: {
      ...tactic, argumentContexts: tactic.argumentContexts.map(({causalVerificationAvailable: _, ...context}) => context),
    }};
    expect(scenarioPredicateSupported(historical, selection, 'hard', {
      kind: 'small_card_big_gain', causalRequired: true,
    })).toBe(true);
  });

  test('matches the published decision and line-universe tuple exactly', () => {
    const purchase = predicateCapability(
      'adjudication.purchase_before_turn_attempt',
      'bid',
      [],
    );
    const continuation = predicateCapability(
      'proof.winning_continuation_line_count',
      'play',
      ['all_legal', 'optimal_strategy'],
    );
    const merged = mergeScenarioPredicateDefinitions([], capability([purchase, continuation]));
    const purchaseDefinition = merged.find((item) => item.kind === purchase.kind)!;
    const continuationDefinition = merged.find((item) => item.kind === continuation.kind)!;
    const base = {
      origin: 'search',
      proofMode: 'exact',
      visibility: 'complete_state',
    } as const;

    expect(scenarioPredicateSupported(
      purchaseDefinition,
      {...base, decisionKind: 'bid'},
      'hard',
    )).toBe(true);
    expect(scenarioPredicateSupported(
      purchaseDefinition,
      {...base, decisionKind: 'play'},
      'hard',
    )).toBe(false);
    expect(scenarioPredicateSupported(purchaseDefinition, base, 'hard')).toBe(false);
    expect(scenarioPredicateSupported(
      continuationDefinition,
      {...base, decisionKind: 'play', lineUniverse: 'optimal_strategy'},
      'hard',
    )).toBe(true);
    expect(scenarioPredicateSupported(
      continuationDefinition,
      {...base, decisionKind: 'play', lineUniverse: 'policy_conditioned'},
      'hard',
    )).toBe(false);
  });

  test('fails closed when a V3 manifest row is missing', () => {
    const merged = mergeScenarioPredicateDefinitions([], capability(undefined));
    const selection = {
      origin: 'search',
      decisionKind: 'bid',
      proofMode: 'exact',
      visibility: 'complete_state',
    } as const;

    expect(merged.every((item) => item.v3ExecutionCapability === null)).toBe(true);
    expect(unsupportedScenarioPredicateIds(
      merged,
      [{predicateId: 'PRED-1', kind: merged[0]!.kind, hardness: 'hard'}],
      selection,
    )).toEqual(['PRED-1']);
  });

  test('filters tactic subtypes from governed decision and utility prerequisites', () => {
    const tactic = predicateCapability('tactic.applicable', 'play', []);
    tactic.decisionKinds = ['bid', 'project', 'play'];
    tactic.executionContexts = (['search', 'exact_trace_prefix'] as const).flatMap((origin) =>
      (['bid', 'project', 'play'] as const).flatMap((decisionKind) => [
        ...(['exact', 'bounded', 'policy_conditioned'] as const).map((proofMode) => ({
          origin, decisionKind, proofMode,
          visibility: 'complete_state' as const, lineUniverses: [],
        })),
        {origin, decisionKind, proofMode: 'exact' as const,
          visibility: 'actor_information_set' as const, lineUniverses: []},
      ]));
    tactic.argumentContexts = [
      {
        argumentName: 'kind',
        argumentValue: 'project_bidding',
        decisionKinds: ['project'],
        requiredUtility: 'awarded_point_margin',
        requiredProofBindings: [{binding: 'root_project_consequence_variables', minimumCount: 2}],
        executionContexts: (['search', 'exact_trace_prefix'] as const).flatMap((origin) =>
          (['exact'] as const).map((proofMode) => ({
            origin, decisionKind: 'project' as const, proofMode,
            visibility: 'complete_state' as const, lineUniverses: [],
          }))),
      },
      {
        argumentName: 'kind',
        argumentValue: 'contract_rescue',
        decisionKinds: ['bid', 'project', 'play'],
        requiredUtility: 'contract_success',
        requiredProofBindings: [{binding: 'root_action_graph', minimumCount: 2}],
      },
      ...(['kabout_creation', 'kabout_preservation', 'kabout_prevention'] as const)
        .map((argumentValue) => ({
          argumentName: 'kind',
          argumentValue,
          decisionKinds: ['bid', 'project', 'play'] as const,
          requiredUtility: 'kabout' as const,
          requiredProofBindings: [{binding: 'base_trick_variables', minimumCount: 2}],
        })),
      ...([
        'entry_unblocking',
        'partner_enablement',
        'trump_control',
        'small_card_big_gain',
        'large_card_sacrifice',
      ] as const).map((argumentValue) => ({
        argumentName: 'kind',
        argumentValue,
        decisionKinds: ['play'] as const,
        requiredUtility: null,
        requiredProofBindings: [{
          binding: 'principal_variation_has_continuation_variable',
          minimumCount: 1,
        }],
      })),
    ];
    const merged = mergeScenarioPredicateDefinitions([], capability([tactic]));
    const definitionValue = merged.find((item) => item.kind === 'tactic.applicable')!;
    const base = {
      origin: 'search',
      proofMode: 'exact',
      visibility: 'complete_state',
    } as const;

    expect(scenarioPredicateArgumentValues(
      definitionValue,
      {...base, decisionKind: 'bid', utility: 'round_win'},
      'kind',
    )).toEqual([]);
    expect(scenarioPredicateArgumentValues(
      definitionValue,
      {...base, decisionKind: 'play', utility: 'contract_success'},
      'kind',
    )).toContain('contract_rescue');
    expect(scenarioPredicateArgumentValues(
      definitionValue,
      {...base, decisionKind: 'play', utility: 'kabout'},
      'kind',
    )).toEqual(expect.arrayContaining([
      'kabout_creation',
      'kabout_preservation',
      'kabout_prevention',
    ]));
    expect(scenarioPredicateSupported(
      definitionValue,
      {...base, decisionKind: 'bid', utility: 'round_win'},
      'hard',
      {kind: 'entry_unblocking'},
    )).toBe(false);
    expect(scenarioPredicateSupported(
      definitionValue,
      {...base, decisionKind: 'project', utility: 'awarded_point_margin'},
      'hard',
      {kind: 'project_bidding'},
    )).toBe(true);
    for (const origin of ['search', 'exact_trace_prefix'] as const) {
      for (const proofMode of ['exact'] as const) {
        expect(scenarioPredicateSupported(definitionValue,
          {...base, origin, proofMode, decisionKind: 'project', utility: 'awarded_point_margin'},
          'hard', {kind: 'project_bidding'})).toBe(true);
      }
    }
    for (const selection of [
      {...base, decisionKind: 'project' as const, utility: 'round_win' as const},
      {...base, decisionKind: 'bid' as const, utility: 'awarded_point_margin' as const},
      {...base, decisionKind: 'project' as const, utility: 'awarded_point_margin' as const,
        proofMode: 'bounded' as const},
      {...base, decisionKind: 'project' as const, utility: 'awarded_point_margin' as const,
        proofMode: 'policy_conditioned' as const},
      {...base, decisionKind: 'project' as const, utility: 'awarded_point_margin' as const,
        visibility: 'actor_information_set' as const},
    ]) {
      expect(scenarioPredicateSupported(definitionValue, selection,
        'hard', {kind: 'project_bidding'})).toBe(false);
      expect(scenarioPredicateArgumentValues(definitionValue, selection, 'kind'))
        .not.toContain('project_bidding');
    }
  });
});
