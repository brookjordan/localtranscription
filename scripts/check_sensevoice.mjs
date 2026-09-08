import fs from 'fs';
const html = fs.readFileSync('results/dashboard.html', 'utf8');
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];
// include everything up to render() (which touches document)
const cut = js.indexOf('function render()');
const core = js.slice(0, cut);
(0, eval)(core + '; globalThis.EXPORTS = { MODELS, CORPUS, FILES, transcriptSegments, scoreModelFile };');
const { MODELS, transcriptSegments, scoreModelFile } = globalThis.EXPORTS;
const sv = MODELS.find(m => m.name.includes('SenseVoice'));
const segs = transcriptSegments(sv, 'antony billington.wav');
console.log('first seg:', JSON.stringify(segs[0]).slice(0, 180));
console.log('segments:', segs.length, '| any ts leak in text:', segs.some(s => /\[\d/.test(s.text)));
console.log('score:', scoreModelFile(sv, 'antony billington.wav').score.accuracy.toFixed(1));
