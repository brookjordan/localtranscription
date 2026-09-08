import fs from 'fs';
let html = fs.readFileSync('results/dashboard.html', 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
let js = m[1];
const marker = '// ---- render';
let core = js.split(marker)[0];
(0, eval)(core + '; globalThis.MODELS = MODELS; globalThis.FILES = FILES;');
const ky = MODELS.find(x => x.name === 'Kyutai STT 2.6B en');
if (!ky.mitigations) { console.log('NO MITIGATIONS'); process.exit(1); }
const REQUIRED = ['baseline', 'chunked', 'vad'];
const BANNED = ['rep-pen', 'temp0.6']; // catastrophic variants removed 2026-09-08
let fail = 0;
for (const file of Object.keys(FILES)) {
  const variants = ky.mitigations[file];
  if (!variants) { console.log('MISSING FILE:', file); fail++; continue; }
  const names = Object.keys(variants);
  for (const banned of BANNED) {
    if (names.includes(banned)) { console.log(`BANNED VARIANT ${banned} still present:`, file); fail++; }
  }
  for (const req of REQUIRED) {
    if (!names.includes(req)) {
      console.log(`MISSING ${req}: ${file} has [${names.join(', ')}]`);
      fail++;
      continue;
    }
    const v = variants[req];
    if (typeof v.accuracy !== 'number' || !v.note || typeof v.text !== 'string') {
      console.log('BAD VARIANT', file, req); fail++;
    }
  }
}
if (fail) process.exit(1);
console.log(`MITIG OK: ${Object.keys(FILES).length} files x ${REQUIRED.join('/')} (+no banned variants)`);
