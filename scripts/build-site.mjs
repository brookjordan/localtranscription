// Build and validate the canonical localtranscription site.
// Usage: node scripts/build-site.mjs [--check]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE = path.join(ROOT, 'site');
const MANIFEST = path.join(SITE, 'assets-manifest.json');
const ASSET_BASE = 'https://assets.brook.dev/localtranscription/';
const CHECK_ONLY = process.argv.includes('--check');

const primaryAssets = [
  'brook.wav',
  'conrad.wav',
  'silly.m4a',
  'sing.mp3',
  'corpus/antony billington.wav',
  'corpus/Balajee.wav',
  'corpus/Brett.wav',
  'corpus/Brook 2.wav',
  'corpus/Dave.wav',
  'corpus/lockpicking lawyer voice.wav',
  "corpus/Mum's voice.wav",
  'corpus/Nana Chris.wav',
  'corpus/Ping.wav',
  'corpus/Pritisman.wav',
  'corpus/Tung.wav',
  'silly.wav',
  'sing.wav',
];

function encodePath(value) {
  return value.split('/').map(encodeURIComponent).join('/');
}

function assetRecord(source, publicationPath) {
  return {
    source,
    path: publicationPath,
    url: ASSET_BASE + encodePath(publicationPath),
  };
}

function buildManifest() {
  const segmentsRoot = path.join(ROOT, 'results/corpus/segments');
  const segmentFiles = fs.existsSync(segmentsRoot)
    ? fs.readdirSync(segmentsRoot, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith('.wav'))
        .map((entry) => {
          const parent = entry.parentPath ?? entry.path;
          const relative = path.relative(segmentsRoot, path.join(parent, entry.name));
          return relative.split(path.sep).join('/');
        })
        .sort()
    : [];

  return {
    schema: 1,
    generated_from: 'results/corpus/segments',
    base_url: ASSET_BASE,
    audio: primaryAssets.map((asset) => assetRecord(`NAS:${asset}`, asset)),
    segments: segmentFiles.map((segment) => assetRecord(`results/corpus/segments/${segment}`, `segments/${segment}`)),
  };
}

function validateSite() {
  const htmlFiles = [];
  for (const entry of fs.readdirSync(SITE, { recursive: true, withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith('.html')) {
      const parent = entry.parentPath ?? entry.path;
      htmlFiles.push(path.join(parent, entry.name));
    }
  }

  const errors = [];
  for (const file of htmlFiles) {
    const html = fs.readFileSync(file, 'utf8');
    if (html.includes('site-assets') || html.includes('file:///') || html.includes('/Users/')) {
      errors.push(`${path.relative(ROOT, file)} contains a local/source-only path`);
    }
    for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
      const ref = match[1];
      if (!ref.startsWith('.') || ref.startsWith('..') === false) continue;
      const candidate = path.resolve(path.dirname(file), ref.split('#')[0].split('?')[0]);
      const isInternalSiteRef = candidate === SITE || candidate.startsWith(`${SITE}${path.sep}`);
      if (isInternalSiteRef && !fs.existsSync(candidate)) {
        errors.push(`${path.relative(ROOT, file)} -> missing ${ref}`);
      }
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
}

const manifest = JSON.stringify(buildManifest(), null, 2) + '\n';
validateSite();

if (CHECK_ONLY) {
  if (!fs.existsSync(MANIFEST) || fs.readFileSync(MANIFEST, 'utf8') !== manifest) {
    throw new Error('site/assets-manifest.json is stale; run npm run site:build');
  }
  console.log(`site check passed (${manifest.match(/"url"/g)?.length ?? 0} published assets)`);
} else {
  fs.writeFileSync(MANIFEST, manifest);
  console.log(`site manifest written: ${path.relative(ROOT, MANIFEST)}`);
  console.log(`site check passed (${manifest.match(/"url"/g)?.length ?? 0} published assets)`);
}
