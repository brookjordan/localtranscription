import { access, constants, mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const audio: string | undefined = process.argv[2];
if (!audio) {
  console.error('Usage: node scripts/transcribe.ts /absolute/path/to/audio');
  process.exit(2);
}

const root: string = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const model: string = resolve(root, 'models/whisper.cpp/tiny.en/ggml-tiny.en.bin');
const results: string = resolve(root, 'results');
const runtime: string = resolve(root, 'vendor/whisper.cpp/build/bin/whisper-cli');
await access(model, constants.R_OK);
await access(runtime, constants.X_OK);
await access(audio, constants.R_OK);
await mkdir(results, { recursive: true });

const prefix: string = resolve(results, `${basename(audio)}.tiny-en`);
const result = spawnSync(runtime, ['-m', model, '-f', audio, '-otxt', '-of', prefix], {
  stdio: 'inherit',
});
process.exit(result.status ?? 1);
