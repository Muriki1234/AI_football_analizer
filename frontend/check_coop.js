import puppeteer from 'puppeteer';
(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.goto('https://ai-football-analizer.vercel.app/upload', { waitUntil: 'networkidle0' });
  const isolated = await page.evaluate(() => window.crossOriginIsolated);
  console.log('crossOriginIsolated:', isolated);
  await browser.close();
  process.exit(0);
})();
