import fs from 'fs';
let html = fs.readFileSync('results/dashboard.html', 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
let js = m[1];
const marker = '// ---- render';
let core = js.includes(marker) ? js.split(marker)[0] : js.slice(0, js.indexOf('document.'));
(0, eval)(core + '; globalThis.MODELS = MODELS; globalThis.CORPUS = CORPUS; globalThis.corpusFor = corpusFor; globalThis.transcriptSegments = transcriptSegments; globalThis.FILES = FILES;');
console.log('MODELS count:', MODELS.length);
const fileKeys = Object.keys(FILES);
console.log('FILES:', fileKeys.join(', '));
let bad = 0;
for (const model of MODELS) {
  for (const f of fileKeys) {
    if (model.failed) continue;
    const segs = transcriptSegments(model, f);
    if (segs === null) continue;
    if (!Array.isArray(segs)) { console.log('BAD', model.name, f); bad++; }
  }
  const speed = model.rtf ? `rtf=${model.rtf}` : 'NO RTF';
  console.log(model.name.padEnd(34), speed);
}
// smoke: score one non-failed model on every file
const par = MODELS.find(x => x.name.includes('Parakeet'));
for (const f of fileKeys) {
  const segs = transcriptSegments(par, f);
  console.log('parakeet', f, segs ? `${segs.length} segs` : 'null');
}
console.log(bad === 0 ? 'HARNESS OK' : `HARNESS BAD: ${bad}`);
