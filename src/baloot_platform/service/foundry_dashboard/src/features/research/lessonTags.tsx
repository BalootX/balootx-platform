import {useId} from 'react';
import type {ResearchLessonTagCatalog, ResearchLessonTagQuery} from './model';

export function toggleLessonTagChoice(catalog: ResearchLessonTagCatalog,
  query: ResearchLessonTagQuery, axis: keyof ResearchLessonTagQuery,
  value: string, checked: boolean): ResearchLessonTagQuery {
  const definition = catalog.axes.find((item) => item.id === axis);
  if (!definition?.options.some((option) => option.value === value)) {
    throw new Error('Unknown lesson tag choice');
  }
  const selected = new Set(query[axis] ?? []);
  if (checked) selected.add(value); else selected.delete(value);
  const values = definition.options.filter((option) => selected.has(option.value))
    .map((option) => option.value);
  const next = {...query};
  if (values.length) next[axis] = values; else delete next[axis];
  return next;
}

export function appendLessonTagQuery(parameters: URLSearchParams,
  catalog: ResearchLessonTagCatalog, query: ResearchLessonTagQuery): void {
  if (Object.keys(query).some((axis) => !catalog.axes.some((item) => item.id === axis))) {
    throw new Error('Unknown lesson tag axis');
  }
  for (const axis of catalog.axes) {
    const values = query[axis.id] ?? [];
    if (new Set(values).size !== values.length
      || values.some((value) => !axis.options.some((option) => option.value === value))) {
      throw new Error('Invalid lesson tag query');
    }
    for (const option of axis.options) {
      if (values.includes(option.value)) parameters.append(axis.queryParameter, option.value);
    }
  }
}

export function lessonTagCatalogReady(catalog: ResearchLessonTagCatalog | undefined,
  hasCopy: (key: string) => boolean): catalog is ResearchLessonTagCatalog {
  return catalog?.schemaVersion === 1 && /^[0-9a-f]{64}$/.test(catalog.sha256)
    && Array.isArray(catalog.axes) && catalog.axes.length > 0
    && catalog.axes.every((axis) => axis && typeof axis.id === 'string'
      && !Object.prototype.hasOwnProperty.call(Object.prototype, axis.id)
      && typeof axis.queryParameter === 'string' && /^lesson[A-Z][A-Za-z]*$/.test(axis.queryParameter)
      && typeof axis.labelKey === 'string' && hasCopy(axis.labelKey)
      && Array.isArray(axis.options) && axis.options.length > 0
      && axis.options.every((option) => option && typeof option.value === 'string'
        && typeof option.labelKey === 'string' && hasCopy(option.labelKey))
      && new Set(axis.options.map((option) => option.value)).size === axis.options.length)
    && new Set(catalog.axes.map((axis) => axis.id)).size === catalog.axes.length
    && new Set(catalog.axes.map((axis) => axis.queryParameter)).size === catalog.axes.length;
}

export function lessonTagResultLabels(catalog: ResearchLessonTagCatalog, tags: Record<string, unknown>): string[] {
  return catalog.axes.flatMap((axis) => {
    const values = tags[axis.id];
    return Array.isArray(values)
      ? axis.options.filter((option) => values.includes(option.value)).map((option) => option.labelKey)
      : [];
  });
}

export function LessonTagQueryControl({catalog, query, onChange, label, help, message}: {
  catalog: ResearchLessonTagCatalog; query: ResearchLessonTagQuery;
  onChange: (query: ResearchLessonTagQuery) => void; label: string; help: string;
  message: (key: string) => string;
}) {
  const helpId = useId();
  return <details className="scenario-card-selector">
    <summary>{label}</summary>
    <p id={helpId} className="scenario-card-selector__help">{help}</p>
    <div className="scenario-grid">
      {catalog.axes.map((axis) => <fieldset key={axis.id} className="scenario-card-selector__group"
        data-lesson-tag-axis={axis.id} aria-describedby={helpId}>
        <legend>{message(axis.labelKey)}</legend>
        <div className="scenario-card-selector__choices">{axis.options.map((option) => <label key={option.value}>
          <input type="checkbox" value={option.value} name={axis.queryParameter}
            checked={(query[axis.id] ?? []).includes(option.value)}
            onChange={(event) => onChange(toggleLessonTagChoice(catalog, query, axis.id, option.value, event.target.checked))}/>
          <span>{message(option.labelKey)}</span>
        </label>)}</div>
      </fieldset>)}
    </div>
  </details>;
}
