import puppeteer from 'puppeteer';
(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  await page.goto('http://localhost:5176/');
  await new Promise(r => setTimeout(r, 2000));
  await browser.close();
  process.exit(0);
})();
