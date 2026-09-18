import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', protocolTimeout: 120000 });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

  console.log('Navigating to http://localhost:5174/upload...');
  await page.goto('http://localhost:5174/upload', { waitUntil: 'networkidle0' });

  // Upload file to the input element!
  console.log('Uploading file...');
  const inputUploadHandle = await page.$('input[type=file]');
  
  // 10MB dummy file
  import('fs').then(fs => fs.writeFileSync('dummy.mp4', Buffer.alloc(10 * 1024 * 1024, 'a')));
  
  await inputUploadHandle.uploadFile('dummy.mp4');
  
  // Wait for 20 seconds to see logs
  await new Promise(r => setTimeout(r, 20000));
  
  await browser.close();
  process.exit(0);
})();
