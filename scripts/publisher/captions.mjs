import assert from 'node:assert/strict';
import {sha} from './policy.mjs';

export function approvedCaption(pool, binding) {
  assert.equal(pool.status, 'APPROVED_CAPTION_POOL', 'APPROVED_POOL_REQUIRED');
  assert(Array.isArray(pool.captions) && pool.captions.length > 0, 'CAPTION_POOL_REQUIRED');
  assert.equal(new Set(pool.captions.map(row => row.index)).size, pool.captions.length, 'DUPLICATE_POOL_INDEX');
  const row = pool.captions.find(row => row.index === binding.captionPoolIndex);
  assert(row && typeof row.caption === 'string' && row.caption.length > 0, 'APPROVED_CAPTION_REQUIRED');
  assert.equal(sha(row.caption), binding.captionSha256, 'CAPTION_SHA');
  assert.equal(row.caption.length, binding.captionCharacters, 'CAPTION_CHARACTERS');
  assert.equal(Buffer.byteLength(row.caption), binding.captionUtf8Bytes, 'CAPTION_UTF8_BYTES');
  assert(binding.platforms.every(platform => row.platforms.includes(platform)), 'CAPTION_PLATFORMS');
  return row.caption;
}
