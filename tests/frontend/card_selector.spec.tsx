import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import {describe, expect, test} from 'vitest';

import type {JsonObject, JsonValue} from '../../src/baloot_platform/service/foundry_dashboard/src/shared/json.js';
import {
  CardSelectorV1Control,
  changeCardSelectorOwnerRelation,
  isCardSelectorSchema,
  toggleCardSelectorChoice,
} from '../../src/baloot_platform/service/foundry_dashboard/src/features/research/cardSelector.js';

const dashboardRequire = createRequire(new URL(
  '../../src/baloot_platform/service/foundry_dashboard/package.json',
  import.meta.url,
));
const {createElement} = dashboardRequire('react') as typeof import('react');
const {renderToStaticMarkup} = dashboardRequire('react-dom/server') as typeof import('react-dom/server');

const ranks = ['seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace'];
const suits = ['spades', 'hearts', 'diamonds', 'clubs'];
const seats = ['north', 'west', 'south', 'east'];
const teams = ['north_south', 'west_east'];
const ownerRelations = ['actor', 'partner', 'opponent_team', 'seat', 'team'];

function localizedEnum(values: string[], key: string): JsonObject {
  return {
    type: 'string',
    enum: values,
    'x-messageArKey': key,
  };
}

const cardSelectorSchema: JsonObject = {
  type: 'object',
  additionalProperties: false,
  'x-messageArKey': 'research_scenario_argument_card_selector',
  properties: {
    cards: {
      type: 'array',
      items: {type: 'integer', minimum: 0, maximum: 31},
      minItems: 1,
      maxItems: 32,
      uniqueItems: true,
      'x-messageArKey': 'research_scenario_argument_cards',
    },
    ranks: {
      type: 'array',
      items: localizedEnum(ranks, 'research_scenario_argument_rank'),
      minItems: 1,
      maxItems: ranks.length,
      uniqueItems: true,
      'x-messageArKey': 'research_scenario_argument_rank',
    },
    suits: {
      type: 'array',
      items: localizedEnum(suits, 'research_scenario_argument_suit'),
      minItems: 1,
      maxItems: suits.length,
      uniqueItems: true,
      'x-messageArKey': 'research_scenario_argument_suit',
    },
    ekka: localizedEnum(
      ['any', 'required', 'forbidden'],
      'research_scenario_argument_ekka',
    ),
    owner: {
      type: 'object',
      additionalProperties: false,
      required: ['relation'],
      'x-messageArKey': 'research_scenario_argument_owner',
      properties: {
        relation: localizedEnum(
          ownerRelations,
          'research_scenario_argument_relation',
        ),
        seat: localizedEnum(seats, 'research_scenario_argument_seat'),
        team: localizedEnum(teams, 'research_scenario_argument_team'),
      },
    },
  },
};

function labelForSchema(schema: JsonObject): string {
  return String(schema['x-messageArKey'] ?? 'missing-label');
}

function labelForOption(value: JsonValue): string {
  return `option-${String(value)}`;
}

describe('CardSelectorV1 visual control', () => {
  test('recognizes only the closed selector schema', () => {
    expect(isCardSelectorSchema(cardSelectorSchema)).toBe(true);
    expect(isCardSelectorSchema({...cardSelectorSchema, additionalProperties: true})).toBe(false);
    expect(isCardSelectorSchema({
      ...cardSelectorSchema,
      properties: {
        ...(cardSelectorSchema.properties as JsonObject),
        oracle: {type: 'boolean'},
      },
    })).toBe(false);
  });

  test('keeps choices in catalog order and clears stale owner fields', () => {
    let value: JsonObject = {};
    value = toggleCardSelectorChoice(value, 'ranks', 'nine', ranks);
    value = toggleCardSelectorChoice(value, 'ranks', 'seven', ranks);
    value = toggleCardSelectorChoice(value, 'cards', 31, [0, 1, 31]);
    value = toggleCardSelectorChoice(value, 'cards', 0, [0, 1, 31]);
    expect(value).toEqual({
      ekka: 'any',
      cards: [0, 31],
      ranks: ['seven', 'nine'],
    });

    value = changeCardSelectorOwnerRelation(
      {...value, owner: {relation: 'seat', seat: 'east'}},
      'team',
      seats,
      teams,
    );
    expect(value.owner).toEqual({relation: 'team', team: 'north_south'});
    expect((value.owner as JsonObject).seat).toBeUndefined();

    value = changeCardSelectorOwnerRelation(value, 'opponent_team', seats, teams);
    expect(value.owner).toEqual({relation: 'opponent_team'});
  });

  test('renders semantic native controls, visible help, and linked errors', () => {
    const markup = renderToStaticMarkup(createElement(CardSelectorV1Control, {
      schema: cardSelectorSchema,
      value: {ranks: ['ace'], ekka: 'required', owner: {relation: 'seat', seat: 'west'}},
      cards: [
        {id: 0, code: 'S7', symbol: '7♠', label: 'سبعة سباتي'},
        {id: 31, code: 'CA', symbol: 'A♣', label: 'آص شيريا'},
      ],
      label: 'محدد الورقة *',
      help: 'اختر ورقة بعينها أو صفاتها.',
      fieldPath: 'PRED-1.arguments.selector',
      errors: [{field: 'ranks', message: 'اختر رتبة صالحة.'}],
      labelForSchema,
      labelForOption,
      onChange: () => undefined,
    }));

    expect(markup).toContain('dir="rtl"');
    expect(markup).toContain('<legend>محدد الورقة *</legend>');
    expect(markup).toContain('اختر ورقة بعينها أو صفاتها.');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).toContain('data-field-path="PRED-1.arguments.selector"');
    expect(markup).toContain('data-field-path="PRED-1.arguments.selector.ranks"');
    expect(markup).toMatch(/href="#card-selector-[^"]+-ranks"/);
    expect(markup).toContain('type="checkbox"');
    expect(markup).toContain('value="0"');
    expect(markup).toContain('value="ace"');
    expect(markup).toContain('value="clubs"');
    expect(markup).toContain('type="radio"');
    expect(markup).toContain('<select>');
    expect(markup).toContain('option-ace');
    expect(markup).toContain('option-west');
    expect(markup).toContain('checked=""');
    expect(markup).not.toContain('<textarea');
    expect(markup).not.toContain('<pre');
  });

  test('keeps touch and RTL layout rules scoped to the selector', async () => {
    const styles = await readFile(new URL(
      '../../src/baloot_platform/service/foundry_dashboard/src/styles.css',
      import.meta.url,
    ), 'utf8');
    expect(styles).toContain('.scenario-card-selector__choices label');
    expect(styles).toContain('min-height:48px');
    expect(styles).toContain('padding-inline:8px');
    expect(styles).toContain('padding-inline-start:22px');
    expect(styles).toContain('.scenario-card-selector:focus-visible');
    expect(styles).toContain('grid-template-columns:repeat(auto-fit,minmax(92px,1fr))');
    expect(styles).toContain('.scenario-arguments>div:has(>.scenario-card-selector){grid-column:1/-1;min-width:0}');
    expect(styles).not.toMatch(/scenario-card-selector__card-grid\{[^}]*repeat\([248],/);
  });
});
