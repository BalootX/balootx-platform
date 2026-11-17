import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {
  configurationEditorEntries,
  configurationEditorObjectArrayAdd,
  configurationEditorObjectArrayRemove,
  configurationEditorObjectArraySchema,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/configuration/model';
import type {JsonObject} from '../../src/baloot_platform/service/foundry_dashboard/src/shared/json';

const schema: JsonObject = {
  type: 'array', minItems: 1, maxItems: 2,
  items: {type: 'object', additionalProperties: false,
    required: ['kind', 'stage', 'budget', 'enabled', 'mode', 'expression'],
    properties: {kind: {const: 'scenario_predicates_v1'}, stage: {enum: ['play']},
      budget: {type: 'integer', default: 999}, enabled: {type: 'boolean', default: true},
      mode: {enum: ['sun', 'hokum']}, expression: {type: 'object', properties: {}}}},
};

test('explicitly added object rows seed only required constants and single choices', () => {
  const frozen = {families: []};
  const entries = configurationEditorEntries({}, {families: schema});
  expect(entries).toEqual([['families', null]]);
  expect(configurationEditorObjectArrayAdd(null, schema)).toEqual([{kind: 'scenario_predicates_v1', stage: 'play'}]);
  expect(configurationEditorObjectArrayAdd(frozen.families, schema)).toEqual([{kind: 'scenario_predicates_v1', stage: 'play'}]);
  expect(frozen).toEqual({families: []});
});

test('add/remove preserve the array and enforce authored item bounds', () => {
  const original = [{kind: 'a'}];
  const added = configurationEditorObjectArrayAdd(original, schema)!;
  expect(original).toEqual([{kind: 'a'}]);
  expect(added).not.toBe(original);
  expect(configurationEditorObjectArrayAdd(added, schema)).toBeUndefined();
  expect(configurationEditorObjectArrayRemove(original, schema, 0)).toBeUndefined();
  expect(configurationEditorObjectArrayRemove(added, schema, 0)).toEqual([{kind: 'scenario_predicates_v1', stage: 'play'}]);
  expect(added).toEqual([{kind: 'a'}, {kind: 'scenario_predicates_v1', stage: 'play'}]);
  expect(configurationEditorObjectArrayRemove(added, schema, -1)).toBeUndefined();
  expect(configurationEditorObjectArrayRemove(added, schema, 2)).toBeUndefined();
  expect(configurationEditorObjectArrayRemove(added, schema, .5)).toBeUndefined();
  expect(configurationEditorObjectArrayRemove([{kind: 'a'}], {...schema, minItems: 0}, 0)).toEqual([]);
});

test('unsupported item schemas and malformed rows retain the existing fallback', () => {
  for (const unsupported of [{type: 'array', items: {type: 'string'}},
    {type: 'array', items: {type: 'object'}},
    {type: 'array', items: {type: 'object', properties: []}},
    {...schema, prefixItems: []}, {type: 'object', items: schema.items}]) {
    expect(configurationEditorObjectArraySchema(unsupported)).toBeUndefined();
    expect(configurationEditorObjectArrayAdd([], unsupported)).toBeUndefined();
  }
  for (const value of [false, ['x'], [null], [[]]]) {
    expect(configurationEditorObjectArrayAdd(value, schema)).toBeUndefined();
    expect(configurationEditorObjectArrayRemove(value, schema, 0)).toBeUndefined();
  }
});

test('visual arrays reuse the schema editor and existing governed native controls', () => {
  const source = readFileSync(new URL('../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url), 'utf8');
  const section = source.slice(source.indexOf('function ConfigObjectArraySection'), source.indexOf('function ConfigObjectSection'));
  expect(section).toContain('className="config-section"');
  expect(section).toContain('<ConfigObjectSection');
  expect(section).toContain('path={[...path,index]}');
  expect(section).toContain('rootSchema={rootSchema}');
  expect(section).toContain('disabled={disabled||!added}');
  expect(section).toContain('items.length>minimum');
  expect(section).toContain('workflow.research_scenario_expression_add_group');
  expect(section).toContain('workflow.research_scenario_remove');
  expect(section).not.toContain('<textarea');
  expect(source.includes('items: resolveSchemaNode(resource, childSchema.items)')).toBe(true);
  expect(source).toContain('value={JSON.stringify(value, null, 2)}');
});

test('ordinary optional scalars expose explicit deletion without changing enum/object removal', () => {
  const source = readFileSync(new URL('../../src/baloot_platform/service/foundry_dashboard/src/main.tsx', import.meta.url), 'utf8');
  const fields = source.slice(source.indexOf('function ConfigField'), source.indexOf('function RawJsonFallback'));
  const label = source.slice(source.indexOf('function FieldLabel'), source.indexOf('function InfoBadge'));
  expect(fields.includes('setOpenHelp={setOpenHelp} onRemove={onRemove} disabled={disabled}><input id={id} disabled={disabled||schema?.const!==undefined}')).toBe(true);
  expect(fields).toMatch(/setOpenHelp=\{setOpenHelp\} onRemove=\{onRemove\} disabled=\{disabled\}><input id=\{id\}[^>]*disabled=\{disabled\} type="number"/);
  expect(fields).toContain('if(!disabled)onRemove();');
  const enumBranch = fields.slice(fields.indexOf('if (options?.length)'), fields.indexOf("if (technicalKey === 'purchaseSuits'"));
  expect(enumBranch).toContain('const clear = onRemove');
  expect(enumBranch).not.toContain('onRemove={onRemove}');
  expect(label).toContain('{onRemove&&<button type="button" className="quiet" disabled={disabled}');
  expect(label).toContain('event.preventDefault();if(!disabled)onRemove();');
  expect(label).toContain('workflow.research_scenario_remove');
  expect(label).toContain("const Container = onRemove ? 'div' : 'label'");
  expect(label).toContain('<label htmlFor={id}>');
});
