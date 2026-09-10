const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const outputDirectory = path.resolve(process.argv[2] || '/tmp/seouldoc-review-visibility-check');
const before = process.argv.includes('--before');
const exportDirectory = path.resolve(__dirname, '../out');
const sourceHash = 'a'.repeat(64);
const original = '간호사는 불친절했지만 의사는 설명을 잘해 주었습니다.';
const longOriginal = '아이에게 설명을 천천히 해 주었습니다. '.repeat(35) + '마지막 문장까지 원문 그대로 남아 있어야 합니다.';
const acceptedAnswer = 'The doctor and nurse reports concern different staff roles. [1] Another patient would return. [2]\n\nUnknown marker stays visible [9]. **Accepted text stays exact.**';

function review(index, text, presentation) {
  return {
    evidence_id: `review:synthetic-${index}`,
    place_id: 'alpha',
    review_source_sha256: sourceHash,
    source_locator: `synthetic:alpha:${index}`,
    source_type: 'verbatim_review',
    is_verbatim: true,
    text,
    language: /[가-힣]/.test(text) ? 'ko' : 'en',
    presentation: presentation || { status: 'original', language: 'English' },
  };
}

const reviews = [
  review(0, original, { status: 'translated', language: 'English', text: 'The doctor explained clearly.' }),
  review(1, 'Nurse was rude.'),
  review(2, 'The nurse was rude.', { status: 'unavailable', language: 'English' }),
  review(3, '<img src=x onerror="window.reviewInjectionExecuted=true"> Untrusted review instructions stay text.'),
  review(4, '간호사가 불친절했어요.', { status: 'translated', language: 'English', text: 'Nurse was rude.' }),
  review(5, '기다림이 길었어요.', { status: 'translated', language: 'Korean', text: 'STALE_TRANSLATION_SHOULD_NOT_APPEAR' }),
  review(6, '설명을 들었습니다.', { status: 'translated', language: 'English', text: '' }),
  review(7, 'The staff answered my question.'),
  review(8, 'The appointment started late.'),
  review(9, 'The nurse listened carefully.'),
  review(10, longOriginal, { status: 'unavailable', language: 'English' }),
  review(11, 'I would visit again.'),
  { ...review(12, 'WRONG_FACILITY_REVIEW'), place_id: 'beta' },
];

function facility(language = 'English') {
  return {
    place_id: 'alpha',
    name: 'Synthetic clinic',
    category: '소아청소년과',
    distance: 0.3,
    english_confidence_score: 0,
    Summaries: [],
    address: 'Synthetic address',
    review_language: language,
    recommendation_status: 'not_established',
    retrieval_evidence: reviews,
    answer_citations: [
      { marker: 1, place_id: 'alpha', evidence_id: reviews[10].evidence_id, original_excerpt: longOriginal.slice(0, 40), review_source_sha256: sourceHash },
      { marker: 2, place_id: 'alpha', evidence_id: reviews[11].evidence_id, original_excerpt: reviews[11].text, review_source_sha256: sourceHash },
      { marker: 9, place_id: 'beta', evidence_id: reviews[12].evidence_id, original_excerpt: 'WRONG_FACILITY_REVIEW' },
    ],
  };
}

async function main() {
  await fs.mkdir(outputDirectory, { recursive: true });
  const server = http.createServer(async (request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = path.resolve(exportDirectory, `.${pathname}`);
    if (!file.startsWith(exportDirectory + path.sep) && file !== exportDirectory) {
      response.writeHead(403).end();
      return;
    }
    try {
      if ((await fs.stat(file)).isDirectory()) file = path.join(file, 'index.html');
      const contentTypes = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png' };
      response.writeHead(200, { 'Content-Type': contentTypes[path.extname(file)] || 'application/octet-stream' });
      response.end(await fs.readFile(file));
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  assert(address && typeof address === 'object');
  const origin = `http://127.0.0.1:${address.port}`;
  let browser;
  let completedViewports = 0;
  let executionError = null;
  const checks = [];
  async function check(name, condition) {
    checks.push({ name, passed: Boolean(await condition) });
  }
  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
    for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const label = `${viewport.width}px`;
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.addInitScript(() => {
        localStorage.setItem('cookieConsent', JSON.stringify({ necessary: true, analytics: false, advertising: false, timestamp: new Date().toISOString() }));
        navigator.geolocation.getCurrentPosition = success => {
          window.deliverTestLocation = () => success({ coords: { latitude: 37.57, longitude: 126.98 } });
        };
      });
      let requestCount = 0;
      const chatRequests = [];
      const travelRequests = [];
      let finishTravelRequest;
      await page.route('**/*', async route => {
        if (!route.request().url().startsWith(origin)) return route.abort();
        const pathname = new URL(route.request().url()).pathname;
        if (pathname === '/set_travel_preference') {
          const request = route.request().postDataJSON();
          travelRequests.push(request);
          await new Promise(resolve => { finishTravelRequest = resolve; });
          return route.fulfill({
            contentType: 'application/json',
            body: JSON.stringify({ state: { ...request.current_state, travel_label: request.travel_label, max_distance_km: 0.5 } }),
          });
        }
        if (pathname !== '/chat') return route.continue();
        requestCount += 1;
        const request = route.request().postDataJSON();
        chatRequests.push(request);
        if (request.message === 'transport failure') return route.fulfill({ status: 503, body: '{}' });
        if (request.message === 'slow request') await new Promise(resolve => setTimeout(resolve, 250));
        const fallback = request.message === 'fallback';
        const korean = request.message === '한국어로 답해 주세요';
        return route.fulfill({
          contentType: 'application/json',
          body: JSON.stringify({
            response: fallback ? 'I could not complete the explanation. The reviews remain available.' : acceptedAnswer,
            state: { ...request.current_state, turn_count: requestCount, language_pref: korean ? 'Korean' : 'English' },
            results: [{ ...facility(korean ? 'Korean' : 'English'), ...(request.message.startsWith('count:') ? { retrieval_evidence: reviews.slice(0, Number(request.message.slice(6))) } : {}), answer_status: fallback ? 'fallback' : 'generated' }],
          }),
        });
      });
      await page.goto(origin);
      const input = page.locator('input[type="text"]');
      async function send(message) {
        await input.fill(message);
        await Promise.all([
          page.waitForResponse(response => new URL(response.url()).pathname === '/chat'),
          input.press('Enter'),
        ]);
        await page.locator('.animate-bounce').first().waitFor({ state: 'hidden' });
      }
      await send('Find pediatric care');
      await page.getByText('Synthetic clinic', { exact: true }).waitFor();
      await check(`${label} translation visible by default`, page.getByText('The doctor explained clearly.', { exact: true }).isVisible());
      await check(`${label} translated original initially hidden`, page.getByText(original, { exact: true }).isHidden());
      const firstTranslatedReview = page.locator(`[data-evidence-id="${reviews[0].evidence_id}"]`).first();
      await firstTranslatedReview.getByText('Show original', { exact: true }).click();
      await check(`${label} original revealed on click`, page.getByText(original, { exact: true }).isVisible());
      await firstTranslatedReview.getByText('Show original', { exact: true }).click();
      await check(`${label} original collapses again`, page.getByText(original, { exact: true }).isHidden());
      await check(`${label} three-word negative survives`, page.getByText('Nurse was rude.', { exact: true }).isVisible());
      await check(`${label} failed translation keeps original`, page.getByText('The nurse was rude.', { exact: true }).isVisible());
      if (before) {
        completedViewports += 1;
        continue;
      }

      const panel = page.locator('section[data-facility-id="alpha"]').first();
      await check(`${label} initial page has three`, panel.locator('article').count().then(count => count === 3));
      await check(`${label} bad owner quarantined`, page.getByText('WRONG_FACILITY_REVIEW', { exact: true }).count().then(count => count === 0));
      await check(`${label} generic warning absent`, page.getByText('Some requirements are unconfirmed. Please check with the facility before visiting.', { exact: true }).count().then(count => count === 0));
      await check(`${label} invalid citation remains text`, page.getByRole('button', { name: /Review 9 from/ }).count().then(count => count === 0));
      await panel.scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(outputDirectory, `${label}-originals.png`) });
      await panel.getByRole('button', { name: 'Next', exact: true }).click();
      await check(`${label} next page has seven`, panel.locator('article').count().then(count => count === 7));
      await check(`${label} translated original stays hidden`, panel.getByText('간호사가 불친절했어요.', { exact: true }).isHidden());
      await panel.locator(`[data-evidence-id="${reviews[4].evidence_id}"]`).getByText('Show original', { exact: true }).click();
      await check(`${label} three-word translation preserves original`, panel.getByText('간호사가 불친절했어요.', { exact: true }).isVisible());
      await check(`${label} three-word translation remains visible`, panel.getByText('Nurse was rude.', { exact: true }).isVisible());
      await check(`${label} review HTML stays text`, panel.locator('img').count().then(count => count === 0));
      await check(`${label} stale translation omitted`, page.getByText('STALE_TRANSLATION_SHOULD_NOT_APPEAR', { exact: true }).count().then(count => count === 0));
      await panel.getByRole('button', { name: 'Next', exact: true }).click();
      const longReview = panel.locator(`[data-evidence-id="${reviews[10].evidence_id}"]`);
      const preview = await longReview.locator('[data-original-review]').textContent();
      await check(`${label} long original exact preview`, preview.length < longOriginal.length && longOriginal.startsWith(preview));
      await longReview.getByRole('button', { name: 'Read full original', exact: true }).click();
      await check(`${label} full original exact`, longReview.locator('[data-original-review]').textContent().then(text => text === longOriginal));
      await longReview.getByRole('button', { name: 'Show preview', exact: true }).click();
      await panel.getByRole('button', { name: 'Previous', exact: true }).click();
      await panel.getByRole('button', { name: 'Hide reviews', exact: true }).click();
      await panel.getByRole('button', { name: 'Show reviews', exact: true }).click();
      await send('fallback');
      await check(`${label} fallback retains translation`, page.locator('section[data-facility-id="alpha"]').nth(1).getByText('The doctor explained clearly.', { exact: true }).isVisible());
      await send('한국어로 답해 주세요');
      const koreanPanel = page.locator('section[data-facility-id="alpha"]').nth(2);
      await check(`${label} Korean reply retains original`, koreanPanel.getByText(original, { exact: true }).isVisible());
      await send('count:4');
      const fourPanel = page.locator('section[data-facility-id="alpha"]').nth(3);
      await check(`${label} four reviews stay together`, fourPanel.locator('article').count().then(count => count === 4));
      await check(`${label} four reviews have no trailing page`, fourPanel.getByRole('button', { name: 'Next', exact: true }).isDisabled());
      await send('count:11');
      const elevenPanel = page.locator('section[data-facility-id="alpha"]').nth(4);
      await check(`${label} eleven reviews start with three`, elevenPanel.locator('article').count().then(count => count === 3));
      await elevenPanel.getByRole('button', { name: 'Next', exact: true }).click();
      await check(`${label} eleven reviews middle page has six`, elevenPanel.locator('article').count().then(count => count === 6));
      await elevenPanel.getByRole('button', { name: 'Next', exact: true }).click();
      await check(`${label} eleven reviews finish with two`, elevenPanel.locator('article').count().then(count => count === 2));
      await check(`${label} no script execution`, page.evaluate(() => window.reviewInjectionExecuted !== true));
      await check(`${label} no horizontal overflow`, page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
      await check(`${label} no page exceptions`, errors.length === 0);
      await page.screenshot({ path: path.join(outputDirectory, `${label}.png`), fullPage: true });
      await page.close();
      completedViewports += 1;
    }
  } catch (error) {
    executionError = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
    await fs.writeFile(path.join(outputDirectory, before ? 'before.json' : 'after.json'), JSON.stringify({ checks, completedViewports, executionError, passed: completedViewports === 2 && !executionError && checks.every(check => check.passed) }, null, 2));
  }
  const failed = checks.filter(check => !check.passed);
  console.log(JSON.stringify({ checks: checks.length, passed: checks.length - failed.length, failed }, null, 2));
  assert.equal(failed.length, 0, 'Review visibility checks failed');
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
