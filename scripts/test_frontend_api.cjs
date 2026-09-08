const { test, afterEach } = require('node:test');
const assert = require('node:assert/strict');
delete process.env.NEXT_PUBLIC_API_URL;
const { apiFetch, postJson, getApiBases } = require('../frontend/lib/api.ts');
const primary = 'https://valerianfourel-seouldoctor-ncs-retriever.hf.space';
const fallback = 'https://valerianfourel-seouldoctor.hf.space';
const originalFetch = global.fetch;
const body = { response: 'Patient reports helpful staff.', state: { exclusions: ['surgery'] }, results: [] };
const reply = (value = body, status = 200) => new Response(JSON.stringify(value), { status });
afterEach(() => { global.fetch = originalFetch; delete global.window; delete global.localStorage; });
function setup(handler) {
  global.window = { location: { hostname: 'www.seouldoc.io' } };
  global.localStorage = { getItem: () => '{"analytics":false}' };
  const calls = [];
  global.fetch = async (url, init) => { calls.push({ url, init }); return handler(calls.length, init); };
  return calls;
}
test('production and previews use HF; Space and local stay same-origin', () => {
  for (const host of ['seouldoc.io', 'www.seouldoc.io', 'preview.vercel.app']) assert.deepEqual(getApiBases(host), [primary, fallback]);
  for (const host of ['localhost', '127.0.0.1', 'example.hf.space']) assert.deepEqual(getApiBases(host), ['']);
});
test('primary success preserves complete response and avoids fallback', async () => {
  const calls = setup(() => reply());
  assert.deepEqual(await postJson('/chat', { message: 'hi', current_state: {} }), body);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, primary + '/chat');
  assert.equal(calls[0].init.credentials, 'same-origin');
});
for (const failure of ['network', '503', 'invalid-json', 'invalid-state']) {
  test(`${failure} retries original Space with identical state and consent`, async () => {
    const calls = setup(n => {
      if (n === 2) return reply();
      if (failure === 'network') throw new TypeError('network');
      if (failure === '503') return reply({}, 503);
      if (failure === 'invalid-json') return new Response('<html>Starting</html>');
      return reply({ response: 'hi', state: [] });
    });
    assert.deepEqual(await postJson('/chat', { message: 'English please', current_state: body.state }), body);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].url, fallback + '/chat');
    assert.equal(calls[0].init.body, calls[1].init.body);
    assert.equal(calls[1].init.headers.get('X-Cookie-Consent'), '{"analytics":false}');
  });
}
for (const status of [400, 401, 403, 422, 429]) {
  test(`${status} is returned without replay`, async () => {
    const calls = setup(() => reply({ detail: 'Request rejected' }, status));
    await assert.rejects(postJson('/chat', {}), /Request rejected/);
    assert.equal(calls.length, 1);
  });
}
test('travel preference falls back with full state', async () => {
  const calls = setup(n => n === 1 ? reply({}, 502) : reply({ state: body.state }));
  assert.deepEqual(await postJson('/set_travel_preference', { current_state: body.state, travel_label: 'Nearby' }), { state: body.state });
  assert.equal(calls.length, 2);
});
test('unrelated mutations are never replayed', async () => {
  const calls = setup(() => reply({}, 503));
  await assert.rejects(postJson('/consent', {}), /temporarily unavailable/);
  assert.equal(calls.length, 1);
});
test('both failures give a bounded error', async () => {
  const calls = setup(() => reply({}, 503));
  await assert.rejects(postJson('/chat', {}), /temporarily unavailable/);
  assert.equal(calls.length, 2);
});
test('caller abort stops without fallback', async () => {
  const controller = new AbortController();
  const calls = setup((n, init) => new Promise((resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason))));
  const pending = apiFetch('/chat', { signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(calls.length, 1);
});
test('attempt timeout proceeds to fallback', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const calls = setup((n, init) => n === 2 ? reply() : new Promise((resolve, reject) => init.signal.addEventListener('abort', () => reject(init.signal.reason))));
  const pending = postJson('/chat', {});
  t.mock.timers.tick(90_000);
  assert.deepEqual(await pending, body);
  assert.equal(calls.length, 2);
});
test('explicit development override is normalized without replacing production or Space routing', () => {
  const { execFileSync } = require('node:child_process');
  const output = execFileSync(process.execPath, ['-e', `
    const {getApiBases} = require('./frontend/lib/api.ts');
    console.log(JSON.stringify(['localhost','test.vercel.app','www.seouldoc.io','demo.hf.space'].map(getApiBases)));
  `], { cwd: require('node:path').resolve(__dirname, '..'), env: { ...process.env, NEXT_PUBLIC_API_URL: ' https://dev.example/api/// ' } });
  assert.deepEqual(JSON.parse(output), [['https://dev.example/api'], [primary, fallback], [primary, fallback], ['']]);
});
