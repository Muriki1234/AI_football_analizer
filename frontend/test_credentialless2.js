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
          <script>
            console.log('Isolated:', crossOriginIsolated);
            console.log('SAB:', typeof SharedArrayBuffer);
          </script>
        </body>
      </html>
    `);
  }
}).listen(5176, () => console.log('Listening on 5176'));
