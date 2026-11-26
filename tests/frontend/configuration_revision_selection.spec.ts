import {expect, test, vi} from 'vitest';
import {openConfigurationRevision} from '../../src/baloot_platform/service/foundry_dashboard/src/features/configuration/model';

const published = {revisionId: 'published', payload: {minimumGain: 10}};
const chosenDraft = {revisionId: 'draft', payload: {minimumGain: 20.5}};

test.each(['resolves', 'rejects'])('keeps the chosen revision and draft when the older request %s last', async (outcome) => {
  const selection = {current: 0};
  const applyProfile = vi.fn();
  const reportError = vi.fn();
  const older = Promise.withResolvers<typeof published>();
  const newer = Promise.withResolvers<typeof chosenDraft>();
  const autoOpen = openConfigurationRevision(selection, () => older.promise, applyProfile, reportError);
  const pick = openConfigurationRevision(selection, () => newer.promise, applyProfile, reportError);

  newer.resolve(chosenDraft);
  await pick;
  if (outcome === 'resolves') older.resolve(published);
  else older.reject(new Error('Earlier revision failed'));
  await autoOpen;

  expect(applyProfile.mock.calls).toEqual([[chosenDraft, chosenDraft.payload]]);
  expect(applyProfile.mock.calls[0][1]).not.toBe(chosenDraft.payload);
  expect(reportError).not.toHaveBeenCalled();
});

test('ignores a revision and its delayed refresh after the family changes', async () => {
  const selection = {current: 0};
  const applyProfile = vi.fn();
  const reportError = vi.fn();
  const response = Promise.withResolvers<typeof published>();
  const open = openConfigurationRevision(selection, () => response.promise, applyProfile, reportError);
  const oldFamilyIntent = selection.current;
  ++selection.current;

  response.resolve(published);
  await open;
  const fetchRefresh = vi.fn(async () => published);
  await openConfigurationRevision(selection, fetchRefresh, applyProfile, reportError, oldFamilyIntent);

  expect(fetchRefresh).not.toHaveBeenCalled();
  expect(applyProfile).not.toHaveBeenCalled();
  expect(reportError).not.toHaveBeenCalled();
});

test('reports errors from the current revision refresh', async () => {
  const selection = {current: 1};
  const applyProfile = vi.fn();
  const reportError = vi.fn();
  const fetchProfile = async () => { throw new Error('Chosen revision failed'); };

  await openConfigurationRevision(selection, fetchProfile, applyProfile, reportError, selection.current);

  expect(applyProfile).not.toHaveBeenCalled();
  expect(reportError).toHaveBeenCalledWith('Chosen revision failed');
});
