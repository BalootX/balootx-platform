#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { parse } from '@babel/parser';

const cwd = fs.realpathSync(process.cwd());
const requestedSources = process.argv.slice(2);
const sourceArguments = requestedSources.length ? requestedSources : ['src/main.tsx'];
const inputErrors = [];
const sourceRecords = [];
const seenPaths = new Map();

function posixPath(value) {
  return value.split(path.sep).join('/');
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isOutside(relativePath) {
  return relativePath === '..'
    || relativePath.startsWith(`..${path.sep}`)
    || path.isAbsolute(relativePath);
}

function inputLabel(absolutePath, rawValue) {
  const relativePath = path.relative(cwd, absolutePath);
  return isOutside(relativePath) ? posixPath(String(rawValue)) : posixPath(relativePath);
}

for (const rawSource of sourceArguments) {
  const absolutePath = path.resolve(cwd, rawSource);
  const label = inputLabel(absolutePath, rawSource);
  const relativePath = path.relative(cwd, absolutePath);
  if (isOutside(relativePath)) {
    inputErrors.push(`${label}:?: source must be inside ${posixPath(path.basename(cwd))}`);
    continue;
  }
  if (!['.ts', '.tsx'].includes(path.extname(absolutePath).toLowerCase())) {
    inputErrors.push(`${label}:?: source must be a .ts or .tsx authored file`);
    continue;
  }
  if (!fs.existsSync(absolutePath)) {
    inputErrors.push(`${label}:?: source does not exist`);
    continue;
  }
  let stat;
  try {
    stat = fs.statSync(absolutePath);
  } catch (error) {
    inputErrors.push(`${label}:?: source cannot be inspected: ${error.message}`);
    continue;
  }
  if (!stat.isFile()) {
    inputErrors.push(`${label}:?: source must be a file`);
    continue;
  }
  let canonicalPath;
  try {
    canonicalPath = fs.realpathSync(absolutePath);
  } catch (error) {
    inputErrors.push(`${label}:?: source cannot be resolved: ${error.message}`);
    continue;
  }
  const canonicalRelativePath = path.relative(cwd, canonicalPath);
  if (isOutside(canonicalRelativePath)) {
    inputErrors.push(`${label}:?: source resolves outside ${posixPath(path.basename(cwd))}`);
    continue;
  }
  if (!['.ts', '.tsx'].includes(path.extname(canonicalPath).toLowerCase())) {
    inputErrors.push(`${label}:?: resolved source must be a .ts or .tsx authored file`);
    continue;
  }
  const canonicalLabel = posixPath(canonicalRelativePath);
  const identity = process.platform === 'win32' ? canonicalPath.toLowerCase() : canonicalPath;
  if (seenPaths.has(identity)) {
    inputErrors.push(`${canonicalLabel}:?: duplicate source input (already provided as ${seenPaths.get(identity)})`);
    continue;
  }
  seenPaths.set(identity, canonicalLabel);
  sourceRecords.push({ label: canonicalLabel, path: canonicalPath });
}

function failWithDiagnostics(errors) {
  process.stderr.write(`${[...errors].sort().join('\n')}\n`);
  process.exit(2);
}

if (inputErrors.length) failWithDiagnostics(inputErrors);
sourceRecords.sort((left, right) => compareText(left.label, right.label));

const staticKeys = new Set(['locale', 'direction', 'page_title']);
const staticKeyConsumers = new Map();
const guardedStaticKeys = new Set();
const requiredStaticKeys = new Set(staticKeys);
const variantFamilies = new Map();
const usedVariantFamilies = new Map();
const catalogConsumers = new Set();
const copyConsumers = new Set();
const unresolved = [];
const parents = new WeakMap();
let variantRegistry = null;

function sourceLine(label, node, message) {
  unresolved.push(`${label}:${node?.loc?.start.line || '?'}: ${message}`);
}

function unwrapExpression(node) {
  let current = node;
  while (
    current
    && ['TSAsExpression', 'TSTypeAssertion', 'TSSatisfiesExpression', 'ParenthesizedExpression'].includes(current.type)
  ) {
    current = current.expression;
  }
  return current;
}

function stringValues(node) {
  const value = unwrapExpression(node);
  if (value?.type === 'StringLiteral') return [value.value];
  if (value?.type === 'ConditionalExpression') {
    return [...stringValues(value.consequent), ...stringValues(value.alternate)];
  }
  return [];
}

function propertyName(node) {
  if (!node) return null;
  if (node.type === 'Identifier') return node.name;
  if (node.type === 'StringLiteral') return node.value;
  return null;
}

function presenceGuard(node, key) {
  return node?.type === 'CallExpression' && node.callee?.name === 'hasFoundryCopy' && node.arguments[0]?.value === key
    || node?.type === 'LogicalExpression' && node.operator === '&&' && (presenceGuard(node.left, key) || presenceGuard(node.right, key));
}

function guardedByPresence(node, key) {
  for (let current = node, parent = parents.get(current); parent; current = parent, parent = parents.get(current)) {
    if (parent.type === 'IfStatement' && parent.consequent === current && presenceGuard(parent.test, key)) return true;
  }
  return false;
}

function findContainingFunction(node) {
  let current = parents.get(node);
  while (current) {
    if (current.type === 'FunctionDeclaration' && current.id) return current.id.name;
    if (current.type === 'ArrowFunctionExpression' || current.type === 'FunctionExpression') {
      const parent = parents.get(current);
      if (parent?.type === 'VariableDeclarator' && parent.id?.type === 'Identifier') {
        return parent.id.name;
      }
      return '<anonymous>';
    }
    current = parents.get(current);
  }
  return null;
}

function attachParents(node, parent = null) {
  if (!node || typeof node !== 'object' || parents.has(node)) return;
  parents.set(node, parent);
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc' || key === 'start' || key === 'end') continue;
    if (Array.isArray(value)) {
      for (const child of value) attachParents(child, node);
    } else if (value && typeof value === 'object') {
      attachParents(value, node);
    }
  }
}

function registerVariantRegistry(node, sourceLabel) {
  if (variantRegistry) {
    sourceLine(
      sourceLabel,
      node,
      `duplicate foundryCopyVariants registry (first declared at ${variantRegistry.label}:${variantRegistry.line})`,
    );
    return;
  }
  variantRegistry = { label: sourceLabel, line: node.loc?.start.line || '?' };
  const expression = unwrapExpression(node.init);
  if (expression?.type !== 'ObjectExpression') {
    sourceLine(sourceLabel, node, 'foundryCopyVariants must be an object literal');
    return;
  }
  for (const property of expression.properties) {
    if (property.type !== 'ObjectProperty') {
      sourceLine(sourceLabel, property, 'invalid foundryCopyVariants member');
      continue;
    }
    const family = propertyName(property.key);
    const value = unwrapExpression(property.value);
    if (!family || value?.type !== 'ArrayExpression') {
      sourceLine(sourceLabel, property, 'invalid foundryCopyVariants member');
      continue;
    }
    if (variantFamilies.has(family)) {
      const first = variantFamilies.get(family);
      sourceLine(
        sourceLabel,
        property,
        `duplicate variant family ${family} (first declared at ${first.label}:${first.line})`,
      );
      continue;
    }
    const variants = value.elements.flatMap((item) => stringValues(item));
    if (variants.length !== value.elements.length) {
      sourceLine(sourceLabel, property, `dynamic variant registration for ${family}`);
      continue;
    }
    const duplicateVariant = variants.find((variant, index) => variants.indexOf(variant) !== index);
    if (duplicateVariant !== undefined) {
      sourceLine(sourceLabel, property, `duplicate variant ${duplicateVariant} in family ${family}`);
      continue;
    }
    variantFamilies.set(family, {
      label: sourceLabel,
      line: property.loc?.start.line || '?',
      variants: [...variants].sort(),
    });
  }
}

function visit(node, sourceLabel) {
  if (
    node?.type === 'VariableDeclarator'
    && node.id?.type === 'Identifier'
    && node.id.name === 'foundryCopyVariants'
    && node.init
  ) {
    registerVariantRegistry(node, sourceLabel);
  }

  if (node?.type === 'CallExpression' && node.callee?.type === 'Identifier') {
    const callName = node.callee.name;
    if (callName === 'tCatalog') catalogConsumers.add(`${sourceLabel}:tCatalog`);
    if (callName === 'readFoundryCopy') copyConsumers.add(`${sourceLabel}:readFoundryCopy`);
    if (callName === 'operatorCopy') {
      for (const value of node.arguments.length ? stringValues(node.arguments[0]) : []) {
        const key = `workflow.${value}`;
        staticKeys.add(key);
        requiredStaticKeys.add(key);
        if (!staticKeyConsumers.has(key)) staticKeyConsumers.set(key, new Set());
        staticKeyConsumers.get(key).add(sourceLabel);
      }
    }
    if (callName === 't') {
      const containingFunction = findContainingFunction(node);
      if (containingFunction !== 'tVariant') {
        const values = node.arguments.length ? stringValues(node.arguments[0]) : [];
        if (!values.length) sourceLine(sourceLabel, node, 'dynamic t() call');
        for (const value of values) {
          staticKeys.add(value);
          (guardedByPresence(node, value) ? guardedStaticKeys : requiredStaticKeys).add(value);
          if (!staticKeyConsumers.has(value)) staticKeyConsumers.set(value, new Set());
          staticKeyConsumers.get(value).add(sourceLabel);
        }
      }
    }
    if (callName === 'tVariant') {
      const families = node.arguments.length ? stringValues(node.arguments[0]) : [];
      if (!families.length) sourceLine(sourceLabel, node, 'dynamic tVariant() family');
      for (const family of families) {
        if (!usedVariantFamilies.has(family)) {
          usedVariantFamilies.set(family, {
            consumers: new Set(),
            label: sourceLabel,
            line: node.loc?.start.line || '?',
          });
        }
        usedVariantFamilies.get(family).consumers.add(sourceLabel);
      }
    }
  }

  for (const [key, value] of Object.entries(node || {})) {
    if (key === 'loc' || key === 'start' || key === 'end') continue;
    if (Array.isArray(value)) {
      for (const child of value) {
        if (child && typeof child.type === 'string') visit(child, sourceLabel);
      }
    } else if (value && typeof value === 'object' && typeof value.type === 'string') {
      visit(value, sourceLabel);
    }
  }
}

for (const sourceRecord of sourceRecords) {
  let source;
  try {
    source = parse(fs.readFileSync(sourceRecord.path, 'utf8'), {
      sourceType: 'module',
      sourceFilename: sourceRecord.label,
      plugins: ['jsx', 'typescript'],
    });
  } catch (error) {
    unresolved.push(
      `${sourceRecord.label}:${error.loc?.line || '?'}: parse error: ${error.reasonCode || error.message}`,
    );
    continue;
  }
  attachParents(source);
  visit(source, sourceRecord.label);
}

for (const [family, usage] of usedVariantFamilies) {
  if (!variantFamilies.has(family)) {
    unresolved.push(`${usage.label}:${usage.line}: unregistered variant family ${family}`);
  }
}
for (const [family, registration] of variantFamilies) {
  if (!usedVariantFamilies.has(family)) {
    unresolved.push(`${registration.label}:${registration.line}: unused variant family ${family}`);
  }
}
if (unresolved.length) failWithDiagnostics(unresolved);

const dynamicKeyConsumers = new Map();
const dynamicKeys = [];
const outputVariantFamilies = {};
for (const [family, registration] of [...variantFamilies.entries()].sort(([left], [right]) => compareText(left, right))) {
  outputVariantFamilies[family] = registration.variants;
  const consumers = usedVariantFamilies.get(family)?.consumers || new Set();
  for (const variant of registration.variants) {
    const key = `${family}_${variant}`;
    dynamicKeys.push(key);
    dynamicKeyConsumers.set(key, new Set(consumers));
  }
}

function consumerObject(entries) {
  return Object.fromEntries(
    [...entries.entries()]
      .sort(([left], [right]) => compareText(left, right))
      .map(([key, consumers]) => [key, [...consumers].sort()]),
  );
}

process.stdout.write(`${JSON.stringify({
  schemaVersion: 2,
  sources: sourceRecords.map((item) => item.label),
  staticKeys: [...staticKeys].sort(),
  optionalStaticKeys: [...guardedStaticKeys].filter((key) => !requiredStaticKeys.has(key)).sort(),
  dynamicKeys: [...new Set(dynamicKeys)].sort(),
  staticKeyConsumers: consumerObject(staticKeyConsumers),
  dynamicKeyConsumers: consumerObject(dynamicKeyConsumers),
  catalogConsumers: [...catalogConsumers].sort(),
  copyConsumers: [...copyConsumers].sort(),
  variantFamilies: outputVariantFamilies,
}, null, 2)}\n`);
