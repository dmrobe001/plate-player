// Build: bundle src/main.js (and the .wgsl shader files it imports as text)
// with esbuild, then inline the bundle and the stylesheet into src/index.html
// to produce one self-contained file: dist/plate-player.html
//
//   npm run build     build once
//   npm run watch     rebuild whenever anything under src/ changes
import * as esbuild from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { watch } from 'node:fs';
import { execSync } from 'node:child_process';

const OUT = 'dist/plate-player.html';

// Git commit and whether the working tree had uncommitted changes, so a
// record can name the exact code that produced it.
function buildInfo() {
  const git = (args) => execSync(`git ${args}`, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  try {
    return { commit: git('rev-parse HEAD'), dirty: git('status --porcelain').length > 0, built: new Date().toISOString() };
  } catch {
    return { commit: 'unknown', dirty: true, built: new Date().toISOString() };
  }
}

async function build() {
  const started = performance.now();
  const bundle = await esbuild.build({
    entryPoints: ['src/main.js'],
    bundle: true,
    format: 'esm',
    target: 'es2022',
    loader: { '.wgsl': 'text' },
    define: { __BUILD_INFO__: JSON.stringify(buildInfo()) },
    write: false,
    legalComments: 'none',
  });
  const js = bundle.outputFiles[0].text;
  const css = await readFile('src/styles.css', 'utf8');
  const template = await readFile('src/index.html', 'utf8');

  // Function replacers so that "$" sequences in the code are not interpreted.
  const html = template
    .replace('<!-- @inline styles.css -->', () => `<style>\n${css}</style>`)
    .replace('<!-- @inline main.js -->', () => `<script type="module">\n${js}</script>`);

  await mkdir('dist', { recursive: true });
  await writeFile(OUT, html);
  const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
  console.log(`built ${OUT} (${kb} KiB) in ${(performance.now() - started).toFixed(0)} ms`);
}

await build();

if (process.argv.includes('--watch')) {
  let pending = null;
  watch('src', { recursive: true }, () => {
    clearTimeout(pending);
    pending = setTimeout(() => build().catch((e) => console.error(e.message)), 50);
  });
  console.log('watching src/ for changes...');
}
