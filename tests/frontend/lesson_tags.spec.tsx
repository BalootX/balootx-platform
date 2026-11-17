import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import type {ResearchLessonTagCatalog, ResearchLessonTagQuery} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/model';
import {appendLessonTagQuery, LessonTagQueryControl, lessonTagCatalogReady, lessonTagResultLabels, toggleLessonTagChoice} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/lessonTags';

const require = createRequire(new URL('../../src/baloot_platform/service/foundry_dashboard/package.json', import.meta.url));
const {createElement} = require('react') as typeof import('react');
const {renderToStaticMarkup} = require('react-dom/server') as typeof import('react-dom/server');
const catalog: ResearchLessonTagCatalog = {schemaVersion: 1, sha256: 'a'.repeat(64), axes: [
  {id: 'impact', queryParameter: 'lessonImpact', labelKey: 'impact', options: [
    {value: 'round-flip', labelKey: 'flip'}, {value: 'points-only', labelKey: 'points'}]},
  {id: 'technique', queryParameter: 'lessonTechnique', labelKey: 'technique', options: [
    {value: 'guard/protect', labelKey: 'guard'}]},
]};

test('multi-choice filters remain detached, canonical, and omit empty axes', () => {
  const original: ResearchLessonTagQuery = {impact: ['points-only']};
  const selected = toggleLessonTagChoice(catalog, original, 'impact', 'round-flip', true);
  expect(original).toEqual({impact: ['points-only']});
  expect(selected).toEqual({impact: ['round-flip', 'points-only']});
  expect(toggleLessonTagChoice(catalog, {impact: ['round-flip']}, 'impact', 'round-flip', false)).toEqual({});
  expect(() => toggleLessonTagChoice(catalog, {}, 'impact', 'invented', true)).toThrow();
});

test('API query uses repeated OR choices and distinct AND axes, not comma strings', () => {
  const parameters = new URLSearchParams({runCode: 'RUN-EXAMPLE'});
  appendLessonTagQuery(parameters, catalog, {impact: ['points-only', 'round-flip'], technique: ['guard/protect']});
  expect(parameters.getAll('lessonImpact')).toEqual(['round-flip', 'points-only']);
  expect(parameters.getAll('lessonTechnique')).toEqual(['guard/protect']);
  expect(parameters.get('runCode')).toBe('RUN-EXAMPLE');
  expect(() => appendLessonTagQuery(new URLSearchParams(), catalog, {mode: ['sun']})).toThrow();
  expect(() => appendLessonTagQuery(new URLSearchParams(), catalog, {impact: ['round-flip', 'round-flip']})).toThrow();
});

test('profile copy gates controls and result labels use only retained known tags', () => {
  expect(lessonTagCatalogReady(undefined, () => true)).toBe(false);
  for (const malformed of [{schemaVersion: 1, sha256: 'a'.repeat(64)},
    {...catalog, axes: [null]}, {...catalog, axes: [{id: 'impact', queryParameter: 'lessonImpact', labelKey: 'impact'}]},
    {...catalog, axes: [{...catalog.axes[0], options: [null]}]},
    {...catalog, axes: [{...catalog.axes[0], id: 'toString'}]},
    {...catalog, axes: [{...catalog.axes[0], id: '__proto__'}]}]) {
    expect(lessonTagCatalogReady(malformed as ResearchLessonTagCatalog, () => true)).toBe(false);
  }
  expect(lessonTagCatalogReady(catalog, (key) => key !== 'guard')).toBe(false);
  expect(lessonTagCatalogReady(catalog, () => true)).toBe(true);
  expect(lessonTagResultLabels(catalog, {impact: ['round-flip'], requested: ['advanced'], technique: ['invented']})).toEqual(['flip']);
  expect(lessonTagResultLabels(catalog, {})).toEqual([]);
});

test('visual filter uses labelled native checkboxes and semantic axis groups', () => {
  const html = renderToStaticMarkup(createElement(LessonTagQueryControl, {catalog,
    query: {impact: ['round-flip']}, onChange: () => undefined,
    label: 'الدروس', help: 'اختيارات داخل النوع أو، وبين الأنواع معًا', message: (key: string) => key}));
  expect(html).toContain('<details');
  expect(html).toContain('<legend>impact</legend>');
  expect(html).toContain('type="checkbox"');
  expect(html).toContain('name="lessonImpact"');
  expect(html).toContain('aria-describedby=');
  expect(html).toContain('checked=""');
});
