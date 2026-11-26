// scripts/gen-blog-banners.mjs — generate blog banner images via ComfyUI.
// Standalone adaptation of localimagegen's scripts/banners/gen-banners.ts
// (same proven workflow builder + model recipe catalog).
//
// Usage: node scripts/gen-blog-banners.mjs [slug ...]   (no args = all)
// Requires ComfyUI running at http://127.0.0.1:8188.

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COMFY_URL = 'http://127.0.0.1:8188';
const DEST_DIR = path.join(__dirname, '../site-assets/banners');

const NEGATIVE_ANIME =
  'worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, extra digits, fewer digits, extra arms, missing limb, blurry, text, watermark';
const NEGATIVE_REALISTIC =
  'lowres, worst quality, bad quality, bad anatomy, sketch, jpeg artifacts, signature, watermark, artist name, old, oldest';

const WORKFLOWS = {
  juggernaut: {
    model: 'Juggernaut-XI-v11.safetensors',
    good_at: ['natural-language', 'cinematic', 'photoreal', 'concept-art', 'industrial', 'landscape', 'architecture', 'audio'],
    steps: 35,
    cfg: 6,
    sampler: 'dpmpp_2m_sde',
    scheduler: 'karras',
    negative: NEGATIVE_REALISTIC,
  },
  illustrious: {
    model: 'illustriousXL_v2.safetensors',
    good_at: ['danbooru-tags', 'illustration', 'painterly', 'fantasy', 'colorful'],
    steps: 40,
    cfg: 5.0,
    sampler: 'euler_ancestral',
    scheduler: 'normal',
    clipSkip: -2,
    negative: NEGATIVE_REALISTIC,
  },
  noobai: {
    model: 'NoobAI-XL-v1.1.safetensors',
    good_at: ['danbooru-tags', 'anime', 'character', 'chibi'],
    steps: 35,
    cfg: 5.0,
    sampler: 'euler_ancestral',
    scheduler: 'normal',
    negative: NEGATIVE_ANIME,
  },
};

function pickWorkflow(goodFor) {
  let best = WORKFLOWS.juggernaut;
  let bestScore = -1;
  for (const preset of Object.values(WORKFLOWS)) {
    const score = preset.good_at.filter((tag) => goodFor.includes(tag)).length;
    if (score > bestScore) { best = preset; bestScore = score; }
  }
  return best;
}

const BANNERS = [
  {
    slug: 'first-words',
    // The first tiny-en transcription: a lone voice appearing as text in the dark.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'cinematic concept art, a dark recording studio at night, a single vintage microphone in sharp focus in the foreground, glowing white text波形波形 waveform lines of light flowing from the microphone mouthpiece and dissolving into floating luminous letters in the dark air, deep blue and teal palette, one warm amber accent light, moody atmospheric, shallow depth of field, wide 16:9 establishing shot, no people',
    seed: 4101,
  },
  {
    slug: 'the-suitcase-parade',
    // Seven engines, one corpus — the model-parade banner.
    goodFor: ['natural-language', 'cinematic', 'photoreal', 'industrial'],
    positive:
      'cinematic photorealistic wide shot, seven different vintage suitcase-style radio receivers and cassette recorders lined up in a row on a dark wooden table, each one glowing softly with its own coloured indicator light, a single shared microphone standing in front of them, dark moody lab background with soft blue rim lighting, dramatic contrast, product photography, 8k detail, no people',
    seed: 4102,
  },
  {
    slug: 'the-gibberish-chorus',
    // Kyutai collapsing into "iculticulticult" — a choir dissolving into static.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'dark surreal cinematic concept art, a grand concert choir of translucent glass human figures singing, their mouths emitting streams of glowing scrambled letters and static noise that dissolve into digital snow particles, fragments of broken waveform ribbons, deep indigo and electric blue palette, glitch aesthetic, dramatic stage lighting from below, wide 16:9 shot',
    seed: 4103,
  },
  {
    slug: 'one-model-two-runtimes',
    // Nemotron 3.5 Transcribe on MLX Audio NVFP4 and NeMo-Speech.cpp IQ4_NL.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio', 'industrial'],
    positive:
      'cinematic concept art, a dark lab bench at night, two identical glowing audio waveform holograms side by side floating above a single bench, the left waveform thin and fast with a cool white light trail, the right waveform thicker with coloured speaker-band segments in teal amber and violet, subtle split lighting dividing the bench into two halves, deep blue and teal palette, one warm amber accent light, moody atmospheric, wide 16:9 establishing shot, no people, no text',
    seed: 4104,
  },
  {
    slug: 'the-silly-test',
    // The intentionally ridiculous benchmark clip.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'cinematic macro concept art, a vintage studio microphone on a dark stage under a single spotlight, a huge glowing translucent speech bubble above it bursting with abstract confetti light fragments and swirling colour shapes spilling out playfully, chaotic but joyful energy, deep blue and teal palette, one warm amber accent light, moody atmospheric, shallow depth of field, wide 16:9 shot, no people, no letters, no text, no symbols',
    seed: 4160,
  },
  {
    slug: 'when-i-start-to-sing',
    // Speech, noise and singing braided into one recording.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'cinematic concept art, a single backlit human silhouette at a vintage microphone switching between three glowing audio streams that braid together in the dark air: a clean thin speech wave, a burst of noise sparks, and a flowing musical ribbon with soft note shapes, deep blue and teal palette, one warm amber accent light, dramatic side lighting, wide 16:9 shot, no faces, no text',
    seed: 4107,
  },
  {
    slug: 'the-harness-before-the-zoo',
    // The result shape that mattered more than the model choice.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'industrial'],
    positive:
      'cinematic concept art, a clean dark laboratory bench with one small audio cassette and a single sheet of glowing structured schematic lines floating above it in orderly rows, blueprint aesthetic, precise geometric light lines and neat connection points, deep blue and teal palette, one warm amber accent light, minimal, moody, wide 16:9 shot, no people, no readable text',
    seed: 4108,
  },
  {
    slug: 'vad-is-a-lever',
    // Segmentation cutting one waveform into clean pieces.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio', 'industrial'],
    positive:
      'cinematic concept art, one long continuous glowing waveform stretched across a dark frame being cut by precise vertical light blades into clean segments, the cut points glowing brighter than the rest of the wave, small segment markers hovering under each slice, deep blue and teal palette, one warm amber accent light, industrial precision aesthetic, moody atmospheric, wide 16:9 shot, no people, no text',
    seed: 4109,
  },
  {
    slug: 'what-the-transcripts-say',
    // Transcripts kept visible instead of reduced to a winner badge.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'cinematic concept art, a dark archive wall of backlit glass panels, each panel displaying lines of glowing abstract transcript marks rendered as luminous dashes in different colours, one panel highlighted brighter than the rest, deep blue and teal palette with warm amber highlights, moody museum lighting, wide 16:9 shot, no people, no readable words',
    seed: 4110,
  },
  {
    slug: 'the-dashboard-is-a-lab-not-a-product',
    // An evaluation surface with teeth, honestly work-in-progress.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'industrial'],
    positive:
      'cinematic concept art, a workbench of glowing instruments and oscilloscope screens in a dark lab, one large screen showing layered comparison charts as abstract luminous blocks and rows, cables and probes neatly arranged, work-in-progress tools scattered honestly on the bench, deep blue and teal palette, one warm amber accent light, moody atmospheric, wide 16:9 shot, no people, no readable text',
    seed: 4111,
  },
  {
    slug: 'two-outputs-and-a-noisy-input',
    // Voice isolation: one input forked into voice and what it took away.
    goodFor: ['natural-language', 'cinematic', 'concept-art', 'audio'],
    positive:
      'cinematic concept art, a dark recording studio, one glowing microphone feeding two separated light streams into two glass vessels on a bench: one vessel holds a clean bright human voice wave, the other holds swirling dark noise static, the split visualised as a clean fork of light between them, deep blue and teal palette, one warm amber accent light, moody atmospheric, wide 16:9 shot, no people, no text',
    seed: 4112,
  },
];

function buildWorkflow(banner, workflow, seed) {
  const width = banner.width ?? 1280;
  const height = banner.height ?? 512;
  const prefix = `localtranscription/banners/${banner.slug}`;
  const clip = workflow.clipSkip ? ['1b', 0] : ['1', 1];
  return {
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: workflow.model } },
    ...(workflow.clipSkip
      ? { '1b': { class_type: 'CLIPSetLastLayer', inputs: { clip: ['1', 1], stop_at_clip_layer: workflow.clipSkip } } }
      : {}),
    '2': { class_type: 'CLIPTextEncode', inputs: { text: banner.positive, clip } },
    '3': { class_type: 'CLIPTextEncode', inputs: { text: workflow.negative, clip } },
    '4': { class_type: 'EmptyLatentImage', inputs: { width, height, batch_size: 1 } },
    '5': {
      class_type: 'KSampler',
      inputs: {
        seed, steps: workflow.steps, cfg: workflow.cfg,
        sampler_name: workflow.sampler, scheduler: workflow.scheduler, denoise: 1,
        model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0],
      },
    },
    '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
    '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: prefix } },
  };
}

async function waitForPrompt(promptId) {
  for (let i = 0; i < 600; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const res = await fetch(`${COMFY_URL}/history/${promptId}`);
    const history = await res.json();
    const entry = history[promptId];
    if (!entry) continue;
    if (entry.status?.status_str === 'error') throw new Error('ComfyUI reported error: ' + JSON.stringify(entry.status?.messages ?? []).slice(0, 500));
    const outputs = entry.outputs ?? {};
    const files = [];
    for (const out of Object.values(outputs)) {
      for (const img of out.images ?? []) files.push(img);
    }
    if (files.length) return files;
  }
  throw new Error('Timed out waiting for prompt ' + promptId);
}

async function generateBanner(banner) {
  const seed = banner.seed ?? Math.floor(Math.random() * 2 ** 32);
  const workflow = pickWorkflow(banner.goodFor);
  const graph = buildWorkflow(banner, workflow, seed);
  console.log(`[${banner.slug}] -> ${workflow.model} ${banner.width ?? 1280}x${banner.height ?? 512} seed=${seed}`);
  const res = await fetch(`${COMFY_URL}/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: graph }),
  });
  if (!res.ok) throw new Error(`Queue failed: ${res.status} ${await res.text()}`);
  const { prompt_id: promptId } = await res.json();
  const files = await waitForPrompt(promptId);
  const src = files[0];
  const imgUrl = `${COMFY_URL}/view?filename=${encodeURIComponent(src.filename)}&subfolder=${encodeURIComponent(src.subfolder ?? '')}&type=${encodeURIComponent(src.type ?? 'output')}`;
  const out = await fetch(imgUrl);
  if (!out.ok) throw new Error(`Fetch failed: ${out.status}`);
  fs.mkdirSync(DEST_DIR, { recursive: true });
  const dest = path.join(DEST_DIR, `${banner.slug}.png`);
  fs.writeFileSync(dest, Buffer.from(await out.arrayBuffer()));
  console.log(`[${banner.slug}] OK -> ${dest} (${(fs.statSync(dest).size / 1024).toFixed(0)} KB)`);
}

const only = process.argv.slice(2);
const targets = only.length ? BANNERS.filter((b) => only.includes(b.slug)) : BANNERS;
if (targets.length === 0) { console.error('No matching banners for: ' + only.join(', ')); process.exit(1); }
for (const b of targets) {
  try { await generateBanner(b); } catch (e) { console.error(`[${b.slug}] FAILED: ${e.message}`); process.exitCode = 1; }
}
