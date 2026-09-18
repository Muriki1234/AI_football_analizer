import http from 'http';
import fs from 'fs';

http.createServer((req, res) => {
  if (req.url === '/') {
    res.writeHead(200, {
      'Content-Type': 'text/html',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless'
    });
    res.end(`
      <html>
        <body>
          <video id="v" controls src="https://www.w3schools.com/html/mov_bbb.mp4"></video>
          <script>
            const v = document.getElementById('v');
            v.onloadedmetadata = () => console.log('LOADED!');
            v.onerror = (e) => console.log('ERROR!', v.error.message);
          </script>
        </body>
      </html>
    `);
  }
}).listen(5175, () => console.log('Listening on 5175'));
