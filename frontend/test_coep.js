import puppeteer from 'puppeteer';
(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  await page.goto('http://localhost:5174/upload', { waitUntil: 'networkidle0' });

  const result = await page.evaluate(async () => {
    return new Promise((resolve) => {
        const v = document.createElement('video');
        // A generic public mp4 on the internet
        v.src = 'https://www.w3schools.com/html/mov_bbb.mp4';
        v.onloadedmetadata = () => resolve('LOADED METADATA!');
        v.onerror = (e) => resolve('ERROR: ' + v.error.message + ' code ' + v.error.code);
        document.body.appendChild(v);
    });
  });

  console.log('Result:', result);
  await browser.close();
  process.exit(0);
})();
