import { access, constants, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
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
const temporaryDir: string = await mkdtemp(resolve(tmpdir(), 'local-voice-transcription-'));
const normalised: string = resolve(temporaryDir, 'input.wav');
let exitCode: number = 1;
try {
  const decoded = spawnSync('ffmpeg', ['-y', '-v', 'error', '-i', audio, '-ac', '1', '-ar', '16000', normalised], {
    stdio: 'inherit',
  });
  if (decoded.status === 0) {
    const result = spawnSync(runtime, ['-m', model, '-f', normalised, '-otxt', '-of', prefix], {
      stdio: 'inherit',
    });
    exitCode = result.status ?? 1;
  } else {
    exitCode = decoded.status ?? 1;
  }
} finally {
  await rm(temporaryDir, { force: true, recursive: true });
}
process.exit(exitCode);
