import {expect, test} from 'vitest';
import {readFileSync} from 'node:fs';
import {
  configurationEditorEntries, configurationEditorExpressionChoice,
  configurationEditorExpressionChoices, configurationEditorSelectExpression,
  configurationEditorObjectArrayAdd, configurationEditorObjectArraySchema,
  configurationEditorSchema,
  configurationEditorEnumMessageKey,
  configurationEditorArrayItemSchema,
  configurationEditorObjectValue,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/configuration/model';
import type {JsonObject} from '../../src/baloot_platform/service/foundry_dashboard/src/shared/json';

const profile = JSON.parse(readFileSync(new URL('../../schemas/profiles/challenge.schema.json', import.meta.url), 'utf8'));
function findExpression(value: any): JsonObject | undefined {
  if (!value || typeof value !== 'object') return undefined;
  if (value['x-familyExpression'] && value.$defs) return value;
  for (const child of Object.values(value)) {
    const result = findExpression(child);
    if (result) return result;
  }
}
const schema = (findExpression(profile)!.$defs as JsonObject).expression as JsonObject;

test('delayed payoff is optional with explicit unit, numeric bounds and catalog copy', () => {
  const tactical = profile.properties.tactical_detectors;
  const native = tactical.properties.nativeDelayedPayoff;
  expect(tactical.required).not.toContain('nativeDelayedPayoff');
  expect(configurationEditorEntries({}, tactical.properties, tactical)).toContainEqual(['nativeDelayedPayoff', null]);
  expect(configurationEditorObjectValue(native)).toEqual({});
  expect(native['x-messageArKey']).toBe('research_native_delayed_payoff_label');
  expect(native['x-helpArKey']).toBe('research_native_delayed_payoff_scope');
  for (const unit of ['raw_card_points', 'awarded_point_margin']) {
    expect(configurationEditorEnumMessageKey(unit, native.properties.gainMetric)).toMatch(/^research_native_delayed_payoff_/);
  }
  expect(native.properties.minimumGain).toMatchObject({type:'number',minimum:0,maximum:100});
  expect(native.properties.minimumGain).not.toHaveProperty('default');
  expect(configurationEditorSchema({nativeDelayedPayoff:{gainMetric:'raw_card_points',minimumGain:20.5}}, tactical)?.properties?.nativeDelayedPayoff).toEqual(native);
});

test('project lifecycle labels use the owning argument-schema localization', () => {
  const choice: any = configurationEditorExpressionChoices(schema).find((branch: any)=>branch.properties.kind?.const==='project.count');
  const state = choice.properties.arguments.properties.state;
  for (const value of ['eligible','declared','comparison_winner','awarded']) {
    expect(configurationEditorEnumMessageKey(value,state)).toBe(state['x-enumMessageArKeys'][value]);
    expect(configurationEditorEnumMessageKey(value,state)).toMatch(/^research_scenario_option_project_/);
  }
  expect(configurationEditorEnumMessageKey('toString',state)).toBeUndefined();
  const allowedChoice: any = configurationEditorExpressionChoices(schema).find((branch: any)=>branch.properties.kind?.const==='deal.card_allowed');
  const seats = allowedChoice.properties.arguments.properties.seats;
  const item = configurationEditorArrayItemSchema(seats);
  const literal: any = (item.anyOf as JsonObject[])[0];
  expect(configurationEditorEnumMessageKey('west', literal)).toBe(seats['x-enumMessageArKeys'].west);
  expect(configurationEditorEnumMessageKey('west', literal)).toBeTruthy();
  expect(seats.items.anyOf[0]).not.toHaveProperty('x-enumMessageArKeys');
});

test('published schema supplies union rows and explicit selection changes only the draft', () => {
  const choices = configurationEditorExpressionChoices(schema);
  const group = {op: 'all', children: [{kind: 'old'}], sha256: 'old'};
  const index = choices.findIndex((branch: any) => branch.properties.kind?.const === 'deal.remaining_suit_count');
  const chosen = configurationEditorSelectExpression(group, schema, index)!;
  expect(group).toEqual({op: 'all', children: [{kind: 'old'}], sha256: 'old'});
  expect(chosen).toEqual({kind: 'deal.remaining_suit_count', version: 1});
  expect(configurationEditorExpressionChoice(chosen, schema)).toBe(index);
  expect(configurationEditorSelectExpression(chosen, schema, index)).toBe(chosen);
  const resolved: any = configurationEditorSchema(chosen, schema);
  expect(resolved.properties.arguments.type).toBe('object');
  expect(configurationEditorEntries(chosen, schema.properties as JsonObject, schema).map(([key])=>key)).toContain('arguments');
  expect(configurationEditorSelectExpression(chosen, schema, -1)).toBeUndefined();
  expect(configurationEditorSelectExpression(chosen, schema, .5)).toBeUndefined();
});

test('unsupported historical predicates remain readable but cannot be newly selected', () => {
  const choices = configurationEditorExpressionChoices(schema);
  const index = choices.findIndex((branch: any) => branch.properties.kind?.const === 'tactic.verified');
  const original = {kind: 'tactic.verified', version: 1, hardness: 'hard', arguments: {kind: 'entry_unblocking'}};
  expect(configurationEditorExpressionChoice(original, schema)).toBe(index);
  expect(configurationEditorSchema(original, schema)?.properties).toHaveProperty('arguments');
  expect(configurationEditorSelectExpression({}, schema, index)).toBeUndefined();
  expect(original.arguments.kind).toBe('entry_unblocking');
});

test.each(['play.action_matches','play.trick_card_count','play.trick_winner_matches'])(
  'native family selector %s has selectable typed arguments', (kind) => {
    const choices = configurationEditorExpressionChoices(schema);
    const index = choices.findIndex((branch: any) => branch.properties.kind?.const === kind);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(choices[index]['x-constructionSupported']).toBe(true);
    const selected = configurationEditorSelectExpression({}, schema, index)!;
    expect(selected).toEqual({kind,version:1});
    const argumentsSchema: any = configurationEditorSchema(selected,schema)?.properties?.arguments;
    expect(argumentsSchema.additionalProperties).toBe(false);
    expect(argumentsSchema.properties.selector.properties.ranks.items['x-familyParameterOptions']).toEqual(['rank']);
    expect(argumentsSchema.properties.selector.properties.owner.properties.seat['x-familyParameterOptions']).toEqual(['seat']);
  });

test('recursive group arrays and missing parameter leaves have visual creation controls', () => {
  const array = {type: 'array', minItems: 1, maxItems: 1, items: schema};
  expect(configurationEditorObjectArraySchema(array)).toBe(schema);
  expect(configurationEditorObjectArrayAdd([], array)).toEqual([{}]);
  expect(configurationEditorObjectArrayAdd([{}], array)).toBeUndefined();
  const parameter = {'x-familyParameterOptions': ['suit'], anyOf: [{type: 'string'}, {type: 'object'}]};
  expect(configurationEditorEntries({}, {suit: parameter, ranks: {type: 'array', items: parameter}})).toEqual([['suit', null], ['ranks', null]]);
});

test('profile UI scopes refs to local resources and uses typed parameter controls, not JSON', () => {
  const source = readFileSync(new URL('../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url), 'utf8');
  const editor = source.slice(source.indexOf('function JsonEditor'), source.indexOf('function ConfigObjectArraySection'));
  expect(editor).toContain('schema?.$id && schema.$defs ? schema : rootSchema');
  expect(editor).toContain('rootSchema={resource}');
  expect(editor).toContain('ConfigParameterArraySection');
  expect(editor).toContain('{$parameter:event.target.value}');
  expect(editor).toContain('x-constructionSupported');
  expect(editor).toContain("hasFoundryCopy('workflow.research_scenario_expression_operator')");
  expect(editor).toContain('configurationEditorExpressionChoices(node).every');
  expect(editor).toContain('expressionCopyAvailable ? node : undefined');
  expect(editor).toContain("hasFoundryCopy('workflow.research_scenario_expression_add_group')");
  expect(editor).toContain("!hasFoundryCopy('workflow.research_lesson_tag_template_source')");
  expect(editor).toContain('schema={alternatives[1] as JsonSchema} rootSchema={null}');
  expect(editor).toContain('onChange={onChange} onRemove={onRemove} openHelp={openHelp}');
  expect(editor).not.toContain('<textarea');
});
