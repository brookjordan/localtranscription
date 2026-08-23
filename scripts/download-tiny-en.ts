import { mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root: string = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const modelDir: string = resolve(root, 'models/whisper.cpp/tiny.en');
await mkdir(modelDir, { recursive: true });

const result = spawnSync(
  'hf',
  ['download', 'ggerganov/whisper.cpp', 'ggml-tiny.en.bin', '--local-dir', modelDir],
  { stdio: 'inherit' },
);
if (result.status !== 0) process.exit(result.status ?? 1);
console.log(`Model ready: ${resolve(modelDir, 'ggml-tiny.en.bin')}`);
