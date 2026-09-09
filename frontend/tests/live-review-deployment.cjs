const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const targetUrl = (process.argv[2] || 'https://www.seouldoc.io').replace(/\/$/, '');
const outputPath = path.resolve(process.argv[3] || '/tmp/seouldoc-live-ui-observation.json');
const applicationRevision = process.argv[4];
const runPath = process.argv[5] && path.resolve(process.argv[5]);
if (!applicationRevision || !runPath) throw new Error('application revision and run path are required');

const hash = value => crypto.createHash('sha256').update(value).digest('hex');

async function main() {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const runRaw = await fs.readFile(runPath, 'utf8');
  const run = JSON.parse(runRaw);
  const scenarioIds = run.selected_scenario_ids;
  if (!Array.isArray(scenarioIds) || scenarioIds.length !== 7) throw new Error('run must select the seven smoke scenarios');
  const cases = new Map((run.cases || []).map(item => [item.id, item]));
  const globalChecks = [];
  const scenarios = [];
  let browser;
  let deployedSource = {};
  const add = (checks, name, passed, details = undefined) => checks.push({ name, passed: Boolean(passed), ...(details === undefined ? {} : { details }) });
  try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
    const request = await browser.newPage();
    const sourceResponse = await request.request.get(`${targetUrl}/ncs-source.json`);
    deployedSource = sourceResponse.ok() ? await sourceResponse.json() : {};
    add(globalChecks, 'deployed source revision matches', deployedSource.branch === 'ncs' && deployedSource.commit === applicationRevision, deployedSource.commit || null);
    await request.close();

    for (const scenarioId of scenarioIds) {
      const scenarioChecks = [];
      const responseRaws = [];
      const caseRecord = cases.get(scenarioId);
      let executionError = null;
      let browserResponse = {};
      const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
      const page = await context.newPage();
      page.on('pageerror', error => add(scenarioChecks, 'no browser page errors', false, error.message));
      await page.addInitScript(() => localStorage.setItem('cookieConsent', JSON.stringify({ necessary: true, analytics: false, advertising: false, timestamp: new Date().toISOString() })));
      try {
        add(scenarioChecks, 'frozen conversation is complete', caseRecord?.status === 'complete');
        const messages = (caseRecord?.turns || []).map(turn => turn.message);
        add(scenarioChecks, 'all frozen user turns are replayable', messages.length > 0 && messages.every(message => typeof message === 'string' && message.trim()));
        await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 90000 });
        const input = page.locator('input[placeholder="Describe what you need..."]');
        for (const message of messages) {
          await input.fill(message);
          const [response] = await Promise.all([
            page.waitForResponse(item => new URL(item.url()).pathname === '/chat', { timeout: 300000 }),
            page.getByRole('button', { name: 'Send message', exact: true }).click(),
          ]);
          const raw = await response.text();
          responseRaws.push(raw);
          try { browserResponse = JSON.parse(raw); } catch { browserResponse = {}; }
          add(scenarioChecks, `turn ${responseRaws.length} returned HTTP success`, response.ok(), response.status());
          add(scenarioChecks, `turn ${responseRaws.length} returned an answer`, typeof browserResponse.response === 'string' && browserResponse.response.trim().length > 0);
          await page.locator('.animate-bounce').first().waitFor({ state: 'hidden', timeout: 300000 }).catch(() => {});
        }
        const cards = Array.isArray(browserResponse.results) ? browserResponse.results : [];
        add(scenarioChecks, 'final turn rendered clinic cards', cards.length > 0, cards.length);
        let originalCount = 0;
        for (const card of cards) {
          const reviews = (Array.isArray(card.retrieval_evidence) ? card.retrieval_evidence : []).filter(review =>
            review && review.place_id === card.place_id && review.source_type === 'verbatim_review'
            && review.is_verbatim === true && typeof review.text === 'string' && review.text.trim()
            && ['translated', 'original', 'unavailable'].includes(review.presentation?.status));
          originalCount += reviews.length;
          const panel = page.locator(`section[data-facility-id="${card.place_id}"]`).last();
          add(scenarioChecks, `clinic ${card.place_id} has one final-turn review panel`, await panel.count() === 1);
          let currentPage = 0;
          for (const review of reviews) {
            let article = panel.locator(`article[data-evidence-id="${review.evidence_id}"]`);
            while (await article.count() === 0 && currentPage < 20) {
              const next = panel.getByRole('button', { name: 'Next', exact: true });
              if (await next.count() === 0 || !await next.isEnabled()) break;
              await next.click();
              currentPage += 1;
              article = panel.locator(`article[data-evidence-id="${review.evidence_id}"]`);
              add(scenarioChecks, `clinic ${card.place_id} later review page is capped at seven`, await panel.locator('article').count() <= 7);
            }
            add(scenarioChecks, `comment ${review.evidence_id} is attached to clinic ${card.place_id}`, await article.count() === 1);
            if (await article.count() !== 1) continue;
            if (review.presentation.status === 'translated' && review.presentation.language === 'English' && review.presentation.text) {
              add(scenarioChecks, `translation ${review.evidence_id} is visible by default`, await article.locator('[data-review-translation]').textContent().then(text => text === review.presentation.text).catch(() => false));
              add(scenarioChecks, `original ${review.evidence_id} is initially collapsed`, await article.locator('[data-original-review]').isHidden().catch(() => false));
              await article.getByText('Show original', { exact: true }).click();
            }
            let original = await article.locator('[data-original-review]').textContent().catch(() => null);
            add(scenarioChecks, `original ${review.evidence_id} has a nonempty exact preview`, typeof original === 'string' && original.length > 0 && review.text.startsWith(original));
            const full = article.getByRole('button', { name: 'Read full original', exact: true });
            if (await full.count()) {
              await full.click();
              original = await article.locator('[data-original-review]').textContent().catch(() => null);
              add(scenarioChecks, `full original ${review.evidence_id} is exact`, original === review.text);
            } else {
              add(scenarioChecks, `original ${review.evidence_id} is exact`, original === review.text);
            }
          }
          const previous = panel.getByRole('button', { name: 'Previous', exact: true });
          while (await previous.count() && await previous.isEnabled()) await previous.click();
          add(scenarioChecks, `clinic ${card.place_id} first review page is capped at three`, await panel.locator('article').count() <= 3);
        }
        add(scenarioChecks, 'final turn exposes at least two original comments', originalCount >= 2, originalCount);
        await page.screenshot({ path: outputPath.replace(/\.json$/, `-${scenarioId}.png`), fullPage: true });
      } catch (error) {
        executionError = { type: error.constructor?.name || 'Error', message: String(error.message || error).slice(0, 1000) };
        add(scenarioChecks, 'live browser replay completed', false, executionError.message);
      } finally {
        await context.close();
      }
      scenarios.push({
        scenario_id: scenarioId,
        status: scenarioChecks.length && scenarioChecks.every(item => item.passed) ? 'passed' : 'failed',
        checks: scenarioChecks,
        response_sha256: hash(responseRaws.join('')),
        response_raws: responseRaws,
        execution_error: executionError,
      });
    }
  } finally {
    if (browser) await browser.close();
  }
  for (const scenario of scenarios) add(globalChecks, `scenario ${scenario.scenario_id} UI replay`, scenario.status === 'passed');
  const observation = {
    schema_version: 1,
    mode: 'live_deployed_ui',
    target_url: targetUrl,
    application_revision: applicationRevision,
    run_file_sha256: hash(runRaw),
    deployed_source: deployedSource,
    scenario_ids: scenarioIds,
    scenarios,
    checks: globalChecks,
  };
  await fs.writeFile(outputPath, JSON.stringify(observation, null, 2) + '\n', { mode: 0o600 });
  const passed = globalChecks.length > 0 && globalChecks.every(item => item.passed);
  process.stdout.write(JSON.stringify({ output: outputPath, passed, scenarios: scenarios.length }) + '\n');
  process.exitCode = passed ? 0 : 1;
}

main().catch(error => {
  process.stderr.write(String(error.stack || error) + '\n');
  process.exitCode = 1;
});
