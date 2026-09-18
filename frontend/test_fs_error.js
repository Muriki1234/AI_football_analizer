import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  // We need cross origin isolated context
  await page.goto('http://localhost:5174/', { waitUntil: 'networkidle0' }).catch(() => {});
  // Wait, if localhost:5174 isn't running, it will fail. Let's restart the serve script.
