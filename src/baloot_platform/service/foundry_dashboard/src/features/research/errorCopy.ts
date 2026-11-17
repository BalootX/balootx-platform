import {t} from '../../shared/localization';
import {researchRecipeErrorCopyKey} from './recipes';

export function researchRecipeErrorMessage(error: unknown): string {
  return researchRecipeErrorCopyKey(error)
    === 'workflow.research_idempotency_unavailable'
    ? t('workflow.research_idempotency_unavailable')
    : t('workflow.m_303534acbecf');
}
