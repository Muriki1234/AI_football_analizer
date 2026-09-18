import puppeteer from 'puppeteer';
(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));

  await page.goto('http://localhost:5174/upload', { waitUntil: 'networkidle0' });

  const result = await page.evaluate(async () => {
    return new Promise((resolve) => {
        const v = document.createElement('video');
        v.crossOrigin = 'anonymous'; // CORS request
        v.src = 'https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.480p.vp9.webm';
        v.onloadedmetadata = () => resolve('LOADED METADATA!');
        v.onerror = (e) => resolve('ERROR: ' + (v.error ? v.error.message + ' code ' + v.error.code : 'unknown'));
        document.body.appendChild(v);
    });
  });

  console.log('Result:', result);
  await browser.close();
  process.exit(0);
})();
