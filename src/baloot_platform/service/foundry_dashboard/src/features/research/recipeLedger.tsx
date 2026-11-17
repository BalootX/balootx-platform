import {useId, useState} from 'react';

import {t} from '../../shared/localization';
import type {ResearchRecipe} from './model';

export type ResearchRecipeLedgerProps = {
  recipes: readonly ResearchRecipe[];
  loadingMore: boolean;
  nextCursor: string | null;
  error: string | null;
  success: string | null;
  onLoadMore: () => void | Promise<void>;
  onRetire: (recipe: ResearchRecipe) => void | Promise<void>;
  retiringCodes: ReadonlySet<string> | readonly string[];
};

function includesCode(
  codes: ResearchRecipeLedgerProps['retiringCodes'],
  code: string,
): boolean {
  const codeSet = codes as ReadonlySet<string>;
  return typeof codeSet.has === 'function'
    ? codeSet.has(code)
    : (codes as readonly string[]).includes(code);
}

export function ResearchRecipeLedger({
  recipes,
  loadingMore,
  nextCursor,
  error,
  success,
  onLoadMore,
  onRetire,
  retiringCodes,
}: ResearchRecipeLedgerProps) {
  const headingId = useId();
  const [retireSuccess, setRetireSuccess] = useState<string | null>(null);
  const idempotencyAvailable =
    typeof window.crypto?.randomUUID === 'function';
  const statusMessage = success || retireSuccess;

  const retire = async (recipe: ResearchRecipe) => {
    if (
      !idempotencyAvailable ||
      recipe.lifecycleStatus !== 'published' ||
      includesCode(retiringCodes, recipe.code) ||
      !window.confirm(t('workflow.research_recipe_retire_confirm'))
    ) {
      return;
    }

    setRetireSuccess(null);
    try {
      await onRetire(recipe);
      setRetireSuccess(t('workflow.research_recipe_retired_success'));
    } catch {
      // The controller owns the governed, operator-facing error message.
    }
  };

  return (
    <section className="panel research-ledger" aria-labelledby={headingId}>
      <header className="panel-heading">
        <div>
          <h3 id={headingId}>{t('workflow.research_recipes_title')}</h3>
          <p>{t('workflow.research_recipes_help')}</p>
        </div>
      </header>

      {error && <p role="alert">{error}</p>}
      {statusMessage && (
        <p role="status" aria-live="polite" aria-atomic="true">
          {statusMessage}
        </p>
      )}
      {!idempotencyAvailable && (
        <p role="alert">{t('workflow.research_idempotency_unavailable')}</p>
      )}

      <div className="bundle-components" role="list">
        {recipes.map((recipe) => {
          const retiring = includesCode(retiringCodes, recipe.code);
          return (
            <div key={recipe.code} role="listitem">
              <strong>{recipe.titleAr}</strong>
              <small dir="ltr">{recipe.code}</small>
              <span>
                {recipe.lifecycleStatus === 'retired' ? (
                  <span className="status neutral">
                    {t('workflow.research_recipe_lifecycle_retired')}
                  </span>
                ) : recipe.lifecycleStatus === 'published' ? (
                  <button
                    type="button"
                    className="secondary"
                    disabled={!idempotencyAvailable || retiring}
                    aria-label={`${t('workflow.research_recipe_retire')} ${recipe.titleAr}`}
                    onClick={() => void retire(recipe)}
                  >
                    {t('workflow.research_recipe_retire')}
                  </button>
                ) : (
                  <span />
                )}
              </span>
            </div>
          );
        })}
      </div>

      {nextCursor && (
        <div className="library-more">
          <button
            type="button"
            className="secondary"
            disabled={loadingMore}
            aria-busy={loadingMore}
            onClick={() => void onLoadMore()}
          >
            {loadingMore
              ? t('workflow.research_recipes_loading_more')
              : t('workflow.research_recipes_load_more')}
          </button>
        </div>
      )}
    </section>
  );
}
