import {requestTimeoutMessage, t} from '../../shared/localization';
import {researchRecipeErrorCopyKey} from './recipes';

export function researchRecipeErrorMessage(error: unknown): string {
  const key = researchRecipeErrorCopyKey(error);
  if (key === 'helpers.request_timeout') return requestTimeoutMessage();
  return key === 'workflow.research_idempotency_unavailable'
    ? t('workflow.research_idempotency_unavailable')
    : t('workflow.m_303534acbecf');
}
