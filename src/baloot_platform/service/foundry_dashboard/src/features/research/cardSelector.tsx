import {useId, useMemo} from 'react';

import type {JsonObject, JsonValue} from '../../shared/json';

export type CardSelectorCardOption = Readonly<{
  id: number;
  code: string;
  symbol: string;
  label: string;
}>;

export type CardSelectorError = Readonly<{
  message: string;
  field?: 'cards' | 'ranks' | 'suits' | 'ekka' | 'owner';
}>;

type CardSelectorState = Readonly<{
  cards: number[];
  ranks: string[];
  suits: string[];
  ekka: string;
  owner?: JsonObject;
}>;

type SelectorListField = 'cards' | 'ranks' | 'suits';

function objectValue(value: JsonValue | undefined): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonObject
    : {};
}

function propertiesOf(schema: JsonObject): JsonObject {
  return objectValue(schema.properties);
}

function enumValues(schema: JsonObject): string[] {
  return Array.isArray(schema.enum)
    ? schema.enum.filter((value): value is string => typeof value === 'string')
    : [];
}

function stringValues(value: JsonValue | undefined): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function numberValues(value: JsonValue | undefined): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === 'number')
    : [];
}

function readSelector(value: JsonValue | undefined): CardSelectorState {
  const source = objectValue(value);
  const owner = objectValue(source.owner);
  return {
    cards: numberValues(source.cards),
    ranks: stringValues(source.ranks),
    suits: stringValues(source.suits),
    ekka: typeof source.ekka === 'string' ? source.ekka : 'any',
    ...(typeof owner.relation === 'string' ? {owner} : {}),
  };
}

function selectorWire(value: CardSelectorState): JsonObject {
  return {
    ekka: value.ekka,
    ...(value.cards.length ? {cards: value.cards} : {}),
    ...(value.ranks.length ? {ranks: value.ranks} : {}),
    ...(value.suits.length ? {suits: value.suits} : {}),
    ...(value.owner ? {owner: value.owner} : {}),
  };
}

function orderedToggle<T extends string | number>(
  current: readonly T[],
  item: T,
  order: readonly T[],
): T[] {
  const selected = new Set(current);
  if (selected.has(item)) selected.delete(item);
  else selected.add(item);
  return order.filter((candidate) => selected.has(candidate));
}

export function toggleCardSelectorChoice(
  value: JsonValue | undefined,
  field: SelectorListField,
  item: string | number,
  order: readonly (string | number)[],
): JsonObject {
  const selector = readSelector(value);
  if (field === 'cards' && typeof item === 'number') {
    return selectorWire({
      ...selector,
      cards: orderedToggle(selector.cards, item, order.filter(
        (candidate): candidate is number => typeof candidate === 'number',
      )),
    });
  }
  if (field !== 'cards' && typeof item === 'string') {
    return selectorWire({
      ...selector,
      [field]: orderedToggle(selector[field], item, order.filter(
        (candidate): candidate is string => typeof candidate === 'string',
      )),
    });
  }
  return selectorWire(selector);
}

export function changeCardSelectorOwnerRelation(
  value: JsonValue | undefined,
  relation: string,
  seats: readonly string[],
  teams: readonly string[],
): JsonObject {
  const selector = readSelector(value);
  if (!relation) return selectorWire({...selector, owner: undefined});
  if (relation === 'seat') {
    return selectorWire({
      ...selector,
      owner: {relation, ...(seats[0] ? {seat: seats[0]} : {})},
    });
  }
  if (relation === 'team') {
    return selectorWire({
      ...selector,
      owner: {relation, ...(teams[0] ? {team: teams[0]} : {})},
    });
  }
  return selectorWire({...selector, owner: {relation}});
}

export function isCardSelectorSchema(schema: JsonObject): boolean {
  if (schema.type !== 'object' || schema.additionalProperties !== false) return false;
  const properties = propertiesOf(schema);
  const names = Object.keys(properties).sort().join(',');
  if (names !== 'cards,ekka,owner,ranks,suits') return false;
  const cards = objectValue(properties.cards);
  const cardItems = objectValue(cards.items);
  const owner = objectValue(properties.owner);
  const ownerProperties = propertiesOf(owner);
  return cards.type === 'array'
    && cardItems.type === 'integer'
    && cardItems.minimum === 0
    && cardItems.maximum === 31
    && objectValue(properties.ranks).type === 'array'
    && objectValue(properties.suits).type === 'array'
    && objectValue(properties.ekka).type === 'string'
    && owner.type === 'object'
    && Object.keys(ownerProperties).sort().join(',') === 'relation,seat,team';
}

export function CardSelectorV1Control({
  schema,
  value,
  cards,
  label,
  help,
  fieldPath,
  errors = [],
  labelForSchema,
  labelForOption,
  onChange,
}: {
  schema: JsonObject;
  value: JsonValue | undefined;
  cards: readonly CardSelectorCardOption[];
  label: string;
  help?: string;
  fieldPath?: string;
  errors?: readonly CardSelectorError[];
  labelForSchema: (schema: JsonObject) => string;
  labelForOption: (value: JsonValue, schema: JsonObject) => string;
  onChange: (value: JsonObject) => void;
}) {
  const generatedId = useId().replaceAll(':', '');
  const baseId = `card-selector-${generatedId}`;
  const helpId = help ? `${baseId}-help` : undefined;
  const errorId = errors.length ? `${baseId}-errors` : undefined;
  const describedBy = [helpId, errorId].filter(Boolean).join(' ') || undefined;
  const selector = readSelector(value);
  const properties = propertiesOf(schema);
  const cardsSchema = objectValue(properties.cards);
  const ranksSchema = objectValue(properties.ranks);
  const rankItemSchema = objectValue(ranksSchema.items);
  const suitsSchema = objectValue(properties.suits);
  const suitItemSchema = objectValue(suitsSchema.items);
  const ekkaSchema = objectValue(properties.ekka);
  const ownerSchema = objectValue(properties.owner);
  const ownerProperties = propertiesOf(ownerSchema);
  const relationSchema = objectValue(ownerProperties.relation);
  const seatSchema = objectValue(ownerProperties.seat);
  const teamSchema = objectValue(ownerProperties.team);
  const ranks = enumValues(rankItemSchema);
  const suits = enumValues(suitItemSchema);
  const ekkaValues = enumValues(ekkaSchema);
  const ownerRelations = enumValues(relationSchema);
  const seats = enumValues(seatSchema);
  const teams = enumValues(teamSchema);
  const ownerRelation = typeof selector.owner?.relation === 'string'
    ? selector.owner.relation
    : '';
  const selectedCardsKey = selector.cards.join(',');
  const cardChoices = useMemo(() => {
    const selected = new Set(
      selectedCardsKey ? selectedCardsKey.split(',').map(Number) : [],
    );
    return cards.map((card) => ({...card, checked: selected.has(card.id)}));
  }, [cards, selectedCardsKey]);
  const fieldTarget = (field?: CardSelectorError['field']) =>
    field ? `${baseId}-${field}` : baseId;

  return <fieldset
    id={baseId}
    className="scenario-card-selector"
    data-field-path={fieldPath}
    aria-describedby={describedBy}
    aria-invalid={errors.length ? true : undefined}
    tabIndex={-1}
    dir="rtl"
  >
    <legend>{label}</legend>
    {help ? <p id={helpId} className="scenario-card-selector__help">{help}</p> : null}
    {errors.length ? <ul id={errorId} className="scenario-card-selector__errors" aria-live="polite">
      {errors.map((error, index) => <li key={`${error.field ?? 'selector'}-${index}`}>
        <a href={`#${fieldTarget(error.field)}`}>{error.message}</a>
      </li>)}
    </ul> : null}

    <fieldset id={`${baseId}-cards`} className="scenario-card-selector__group scenario-card-selector__cards" data-field-path={fieldPath ? `${fieldPath}.cards` : undefined} tabIndex={-1}>
      <legend>{labelForSchema(cardsSchema)}</legend>
      <div className="scenario-card-selector__card-grid">
        {cardChoices.map((card) => <label key={card.id}>
          <input
            type="checkbox"
            name={`${baseId}-cards`}
            value={card.id}
            checked={card.checked}
            onChange={() => onChange(toggleCardSelectorChoice(
              value,
              'cards',
              card.id,
              cards.map((option) => option.id),
            ))}
          />
          <bdi dir="ltr" aria-hidden="true">{card.symbol || card.code}</bdi>
          <span>{card.label}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset id={`${baseId}-ranks`} className="scenario-card-selector__group" data-field-path={fieldPath ? `${fieldPath}.ranks` : undefined} tabIndex={-1}>
      <legend>{labelForSchema(ranksSchema)}</legend>
      <div className="scenario-card-selector__choices">
        {ranks.map((rank) => <label key={rank}>
          <input
            type="checkbox"
            name={`${baseId}-ranks`}
            value={rank}
            checked={selector.ranks.includes(rank)}
            onChange={() => onChange(toggleCardSelectorChoice(value, 'ranks', rank, ranks))}
          />
          <span>{labelForOption(rank, rankItemSchema)}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset id={`${baseId}-suits`} className="scenario-card-selector__group" data-field-path={fieldPath ? `${fieldPath}.suits` : undefined} tabIndex={-1}>
      <legend>{labelForSchema(suitsSchema)}</legend>
      <div className="scenario-card-selector__choices">
        {suits.map((suit) => <label key={suit}>
          <input
            type="checkbox"
            name={`${baseId}-suits`}
            value={suit}
            checked={selector.suits.includes(suit)}
            onChange={() => onChange(toggleCardSelectorChoice(value, 'suits', suit, suits))}
          />
          <span>{labelForOption(suit, suitItemSchema)}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset id={`${baseId}-ekka`} className="scenario-card-selector__group" data-field-path={fieldPath ? `${fieldPath}.ekka` : undefined} tabIndex={-1}>
      <legend>{labelForSchema(ekkaSchema)}</legend>
      <div className="scenario-card-selector__choices">
        {ekkaValues.map((ekka) => <label key={ekka}>
          <input
            type="radio"
            name={`${baseId}-ekka`}
            value={ekka}
            checked={selector.ekka === ekka}
            onChange={() => onChange(selectorWire({...selector, ekka}))}
          />
          <span>{labelForOption(ekka, ekkaSchema)}</span>
        </label>)}
      </div>
    </fieldset>

    <fieldset id={`${baseId}-owner`} className="scenario-card-selector__group scenario-card-selector__owner" data-field-path={fieldPath ? `${fieldPath}.owner` : undefined} tabIndex={-1}>
      <legend>{labelForSchema(ownerSchema)}</legend>
      <label>
        <span>{labelForSchema(relationSchema)}</span>
        <select
          value={ownerRelation}
          onChange={(event) => onChange(changeCardSelectorOwnerRelation(
            value,
            event.target.value,
            seats,
            teams,
          ))}
        >
          <option value="">—</option>
          {ownerRelations.map((relation) => <option key={relation} value={relation}>
            {labelForOption(relation, relationSchema)}
          </option>)}
        </select>
      </label>
      {ownerRelation === 'seat' ? <label>
        <span>{labelForSchema(seatSchema)}</span>
        <select
          value={typeof selector.owner?.seat === 'string' ? selector.owner.seat : ''}
          onChange={(event) => onChange(selectorWire({
            ...selector,
            owner: {relation: 'seat', seat: event.target.value},
          }))}
        >
          {seats.map((seat) => <option key={seat} value={seat}>
            {labelForOption(seat, seatSchema)}
          </option>)}
        </select>
      </label> : null}
      {ownerRelation === 'team' ? <label>
        <span>{labelForSchema(teamSchema)}</span>
        <select
          value={typeof selector.owner?.team === 'string' ? selector.owner.team : ''}
          onChange={(event) => onChange(selectorWire({
            ...selector,
            owner: {relation: 'team', team: event.target.value},
          }))}
        >
          {teams.map((team) => <option key={team} value={team}>
            {labelForOption(team, teamSchema)}
          </option>)}
        </select>
      </label> : null}
    </fieldset>
  </fieldset>;
}
