import { defineConfig } from 'vite';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Dev-only capture endpoint.
 *
 * The scene can only be inspected by rendering it, and a headless/hidden tab
 * never composites, so `computer:screenshot` is unavailable. This lets the page
 * POST a data URL of its own canvas to disk:
 *
 *   fetch('/__shot?name=dawn', { method: 'POST', body: canvas.toDataURL('image/png') })
 *
 * Writes to captures/<name>.png. Dev server only — never part of a build.
 */
function captureEndpoint() {
  return {
    name: 'sakura-capture',
    apply: 'serve',
    configureServer(server) {
      const dir = join(server.config.root, 'captures');
      mkdirSync(dir, { recursive: true });
      server.middlewares.use('/__shot', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          return res.end('POST only');
        }
        const name = (new URL(req.url, 'http://x').searchParams.get('name') || 'shot')
          .replace(/[^a-z0-9._-]/gi, '_');
        let body = '';
        req.setEncoding('utf8');
        req.on('data', (c) => {
          body += c;
        });
        req.on('end', () => {
          try {
            const comma = body.indexOf(',');
            const meta = body.slice(0, comma);
            const ext = meta.includes('jpeg') ? 'jpg' : 'png';
            const buf = Buffer.from(body.slice(comma + 1), 'base64');
            const file = join(dir, `${name}.${ext}`);
            writeFileSync(file, buf);
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ ok: true, file, bytes: buf.length }));
          } catch (err) {
            res.statusCode = 500;
            res.end(JSON.stringify({ ok: false, error: String(err) }));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [captureEndpoint()],
  server: { host: true, port: 5174, open: false },
  build: {
    target: 'esnext',
    sourcemap: true,
    chunkSizeWarningLimit: 2000,
  },
  assetsInclude: ['**/*.glsl', '**/*.vert', '**/*.frag'],
});
