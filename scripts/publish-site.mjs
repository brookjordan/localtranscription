// Publish the canonical site to the website repository.
// Preview: node scripts/publish-site.mjs
// Apply:   node scripts/publish-site.mjs --apply

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE = path.join(ROOT, 'site');
const TARGET = '/Users/brook.jordan/git/brookjordan/brookjordan.github.io/projects/localtranscription';
const APPLY = process.argv.includes('--apply');

const check = spawnSync('node', [path.join(ROOT, 'scripts/build-site.mjs'), '--check'], {
  cwd: ROOT,
  encoding: 'utf8',
});
if (check.status !== 0) {
  process.stdout.write(check.stdout ?? '');
  process.stderr.write(check.stderr ?? '');
  process.exit(check.status ?? 1);
}

const expectedTarget = '/Users/brook.jordan/git/brookjordan/brookjordan.github.io/projects/localtranscription';
if (TARGET !== expectedTarget || !TARGET.endsWith('/projects/localtranscription')) {
  throw new Error(`Refusing unexpected publish target: ${TARGET}`);
}

const args = ['-a', '--delete', '--exclude=.DS_Store', `${SOURCE}/`, `${TARGET}/`];
if (!APPLY) {
  console.log(`publish preview only; rerun with --apply: rsync ${args.join(' ')}`);
  process.exit(0);
}

const result = spawnSync('rsync', args, { cwd: ROOT, encoding: 'utf8' });
process.stdout.write(result.stdout ?? '');
process.stderr.write(result.stderr ?? '');
if (result.status !== 0) process.exit(result.status ?? 1);
console.log(`published ${SOURCE} -> ${TARGET}`);
