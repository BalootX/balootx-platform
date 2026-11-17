import type {JsonObject, JsonValue} from '../../shared/json';

function object(value: JsonValue | undefined): JsonObject {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

export function configurationEditorEnum(schema?: JsonObject): JsonValue[] | undefined {
  if (Array.isArray(schema?.enum)) return schema.enum;
  const alternatives = schema?.anyOf;
  if (!Array.isArray(alternatives) || !alternatives.length
    || !alternatives.every((field) => Object.hasOwn(object(field), 'const'))) {
    return schema?.type === 'boolean' ? [false, true] : undefined;
  }
  return [...new Set(alternatives.map((field) => object(field).const))];
}

export function configurationEditorEnumTitle(value: JsonValue, schema?: JsonObject): string | undefined {
  const choice = Array.isArray(schema?.anyOf) ? schema.anyOf.map(object)
    .find((field) => field.const === value) : undefined;
  return typeof choice?.title === 'string' ? choice.title : undefined;
}

export function configurationEditorEnumMessageKey(value: JsonValue, schema?: JsonObject): string | undefined {
  const keys = object(schema?.['x-enumMessageArKeys']);
  const key = Object.hasOwn(keys, String(value)) ? keys[String(value)] : undefined;
  return typeof key === 'string' ? key : undefined;
}

export function configurationEditorArrayOptions(schema?: JsonObject): JsonValue[] | undefined {
  if (schema?.type !== 'array' || schema.uniqueItems !== true) return undefined;
  const options = configurationEditorEnum(object(schema.items));
  return options?.length && options.every((item) => ['string', 'number', 'boolean'].includes(typeof item))
    ? options : undefined;
}

export function configurationEditorArrayItemSchema(schema: JsonObject): JsonObject {
  const items = object(schema.items);
  const labels = Object.fromEntries(['x-messageArKey', 'x-enumMessageArKeys'].flatMap((key)=>
    Object.hasOwn(schema,key)?[[key,schema[key]]]:[]));
  return {...items,...labels,...(Array.isArray(items.anyOf)?{
    anyOf:items.anyOf.map((branch,index)=>index===0?{...object(branch),...labels}:branch),
  }:{})};
}

export function configurationEditorObjectValue(schema?: JsonObject): JsonObject {
  // Explicit creation may seed only mandatory singleton values, never budgets.
  return Object.fromEntries(Object.entries(object(schema?.properties)).flatMap(([key, definition]) => {
    if (!Array.isArray(schema?.required) || !schema.required.includes(key)) return [];
    const field = object(definition);
    if (Object.hasOwn(field, 'const')) return [[key, field.const]];
    const options = configurationEditorEnum(field);
    return options?.length === 1 ? [[key, options[0]]] : [];
  }));
}

export function configurationEditorObjectArraySchema(schema?: JsonObject): JsonObject | undefined {
  const items = object(schema?.items);
  return schema?.type === 'array' && !Object.hasOwn(schema, 'prefixItems')
    && items.type === 'object' && (items['x-familyExpression'] === true
      || items.properties && typeof items.properties === 'object'
      && !Array.isArray(items.properties)) ? items : undefined;
}

function configurationEditorObjectArrayItems(value: JsonValue | undefined, schema: JsonObject): JsonObject[] | undefined {
  if (!configurationEditorObjectArraySchema(schema)) return undefined;
  if (value == null) return [];
  return Array.isArray(value) && value.every((item) => item && typeof item === 'object' && !Array.isArray(item))
    ? value as JsonObject[] : undefined;
}

export function configurationEditorObjectArrayAdd(value: JsonValue | undefined, schema: JsonObject): JsonObject[] | undefined {
  const items = configurationEditorObjectArrayItems(value, schema);
  if (!items || typeof schema.maxItems === 'number' && items.length >= schema.maxItems) return undefined;
  return [...items, configurationEditorObjectValue(configurationEditorObjectArraySchema(schema))];
}

export function configurationEditorObjectArrayRemove(value: JsonValue | undefined, schema: JsonObject, index: number): JsonObject[] | undefined {
  const items = configurationEditorObjectArrayItems(value, schema);
  const minimum = typeof schema.minItems === 'number' ? schema.minItems : 0;
  if (!items || !Number.isInteger(index) || index < 0 || index >= items.length || items.length <= minimum) return undefined;
  return items.filter((_, itemIndex) => itemIndex !== index);
}

export function configurationEditorChange(value: JsonObject, schema: JsonObject | null | undefined, key: string, child: JsonValue): JsonObject {
  const next = {...value, [key]: child};
  // Only an explicit discriminator edit applies a branch. Merely opening an
  // old profile must not insert fields or change its frozen identity.
  const branches = Array.isArray(schema?.oneOf) ? schema.oneOf.map(object).filter((branch) => {
    const properties = object(branch.properties);
    return Object.hasOwn(object(properties[key]), 'const')
      && Object.entries(properties).every(([name, definition]) => {
        const field = object(definition);
        return !Object.hasOwn(field, 'const') || next[name] === field.const;
      });
  }) : [];
  if (branches.length !== 1) return next;
  const branch = branches[0];
  const properties = {...object(schema?.properties), ...object(branch.properties)};
  for (const name of Array.isArray(branch.required) ? branch.required : []) {
    if (typeof name !== 'string' || Object.hasOwn(next, name)) continue;
    const field = object(properties[name]);
    if (Object.hasOwn(field, 'const')) next[name] = field.const;
    else if (Array.isArray(field.enum) && field.enum.length === 1) next[name] = field.enum[0];
  }
  // not(required:[a,b]) does not forbid either field individually. Only the
  // single-field form can safely remove a field when switching back.
  const negated = object(branch.not);
  for (const clause of [negated, ...(Array.isArray(negated.anyOf) ? negated.anyOf.map(object) : [])]) {
    const forbidden = clause.required;
    if (Object.keys(clause).length === 1 && Array.isArray(forbidden) && forbidden.length === 1 && typeof forbidden[0] === 'string') delete next[forbidden[0]];
  }
  return next;
}

export function configurationEditorExpressionChoices(schema?: JsonObject | null): JsonObject[] {
  return schema?.['x-familyExpression'] === true && Array.isArray(schema.oneOf)
    ? schema.oneOf.map(object).filter((branch) => {
      const key = branch['x-discriminator'];
      return typeof key === 'string' && Object.hasOwn(object(object(branch.properties)[key]), 'const');
    }) : [];
}

export function configurationEditorExpressionChoice(value: JsonObject, schema?: JsonObject | null): number {
  return configurationEditorExpressionChoices(schema).findIndex((branch) => {
    const key = String(branch['x-discriminator']);
    return value[key] === object(object(branch.properties)[key]).const;
  });
}

export function configurationEditorSelectExpression(value: JsonObject, schema: JsonObject, index: number): JsonObject | undefined {
  if (!Number.isInteger(index)) return undefined;
  const branch = configurationEditorExpressionChoices(schema)[index];
  if (!branch || branch['x-constructionSupported'] === false) return undefined;
  if (configurationEditorExpressionChoice(value, schema) === index) return value;
  const properties = object(branch.properties);
  const samePredicate = value.kind === object(properties.kind).const;
  const next = Object.fromEntries(Object.entries(value).filter(([key]) =>
    Object.hasOwn(properties, key) && key !== 'sha256' && (key !== 'arguments' || samePredicate)));
  for (const [key, definition] of Object.entries(properties)) {
    const field = object(definition);
    if (Object.hasOwn(field, 'const')) next[key] = field.const;
  }
  return {...configurationEditorObjectValue(branch), ...next};
}

function configurationEditorBranches(value: JsonObject, schema?: JsonObject | null): JsonObject[] {
  if (schema?.['x-familyExpression'] === true) {
    const selected = configurationEditorExpressionChoice(value, schema);
    return selected < 0 ? [] : [configurationEditorExpressionChoices(schema)[selected]];
  }
  const alternatives = Array.isArray(schema?.oneOf) ? schema.oneOf : schema?.anyOf;
  return Array.isArray(alternatives) ? alternatives.map(object).filter((branch) => {
    const discriminators = Object.entries(object(branch.properties)).filter(([, field]) => Object.hasOwn(object(field), 'const'));
    return discriminators.length > 0 && discriminators.every(([key, field]) => value[key] === object(field).const);
  }) : [];
}

export function configurationEditorSchema(value: JsonObject, schema?: JsonObject | null): JsonObject | undefined {
  if (!schema) return undefined;
  const branches = configurationEditorBranches(value, schema);
  if (branches.length !== 1) return schema;
  const branch = branches[0];
  return {...schema, properties: {...object(schema.properties), ...object(branch.properties)},
    required: [...new Set([...(Array.isArray(schema.required) ? schema.required : []),
      ...(Array.isArray(branch.required) ? branch.required : [])])]};
}

export function configurationEditorEntries(value: JsonObject, properties: JsonObject = {}, schema?: JsonObject | null): [string, JsonValue][] {
  const entries = Object.entries(value);
  let required = schema?.required;
  if (Array.isArray(schema?.oneOf) || Array.isArray(schema?.anyOf)) {
    const branches = configurationEditorBranches(value, schema);
    // Only a uniquely selected branch can advertise missing-field defaults.
    if (branches.length === 1) {
      properties = object(branches[0].properties);
      required = branches[0].required;
    } else if (branches.length || Object.keys(value).length) {
      properties = {};
      required = [];
    }
  }
  // Required primitives stay blank; only authored booleans receive defaults.
  for (const [key, definition] of Object.entries(properties)) {
    if (Object.hasOwn(value, key) || !definition || typeof definition !== 'object' || Array.isArray(definition)) continue;
    if (Array.isArray(definition['x-familyParameterOptions'])) entries.push([key, null]);
    else if (definition.type === 'array' && Array.isArray(object(definition.items)['x-familyParameterOptions'])) entries.push([key, null]);
    else if (definition.type === 'object' && (definition.properties || definition['x-familyExpression'] === true)) entries.push([key, null]);
    else if (configurationEditorArrayOptions(definition)) entries.push([key, null]);
    else if (configurationEditorObjectArraySchema(definition)) entries.push([key, null]);
    else if (definition.type === 'boolean' && typeof definition.default === 'boolean') entries.push([key, definition.default]);
    else if (schema && !Array.isArray(schema.oneOf) && !Array.isArray(schema.anyOf) && definition.type !== 'boolean' && configurationEditorEnum(definition)?.length) entries.push([key, null]);
    else if (Array.isArray(required) && required.includes(key)
      && (['string', 'number', 'integer'].includes(String(definition.type))
        || configurationEditorEnum(definition))) entries.push([key, null]);
  }
  return entries;
}

export type ConfigProfile = {
  revisionId: string;
  family: string;
  profileId: string;
  version: string;
  scope: string;
  scopeKey: string;
  status: string;
  payloadSha256: string;
  sequence: number;
  etag: number;
  updatedAt: string;
  payload?: JsonObject;
};

export type ConfigBundle = {
  revisionId: string;
  bundleId: string;
  version: string;
  status: string;
  mode?: string;
  componentRevisions: Record<string, string>;
  contentSha256: string;
  sequence?: number;
  etag: number;
};

export function configurationPreviewComponents(base: ConfigBundle | undefined, profile: ConfigProfile | null): Record<string, string> | null {
  if (!base || base.status !== 'published' || !profile || !['draft', 'published'].includes(profile.status) || !Object.hasOwn(base.componentRevisions, profile.family)) return null;
  return {...base.componentRevisions, [profile.family]: profile.revisionId};
}
