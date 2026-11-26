import assert from 'node:assert/strict';
import test from 'node:test';

import { BalootClient } from './client.ts';

test('identity SDK sends paging options and retains legacy friends route', async () => {
  const received = [];
  const page = {items: [], nextCursor: 'next-_'};
  const client = new BalootClient({
    baseUrl: 'https://example.test',
    fetchImpl: async input => {
      const url = new URL(input);
      received.push([url.pathname, Object.fromEntries(url.searchParams)]);
      return Response.json(page);
    },
  });

  await client.friends();
  assert.deepEqual(await client.friendsPage({limit: 7, cursor: 'friend-_'}), page);
  assert.deepEqual(await client.friendRequests({cursor: 'request-_'}), page);
  assert.deepEqual(await client.searchUsers('Fixture + %_', {limit: 3, cursor: 'user-_'}), page);
  assert.deepEqual(received, [
    ['/api/v1/friends', {}],
    ['/api/v2/friends', {cursor: 'friend-_', limit: '7'}],
    ['/api/v2/friends/requests', {cursor: 'request-_'}],
    ['/api/v2/users:search', {query: 'Fixture + %_', cursor: 'user-_', limit: '3'}],
  ]);
});
