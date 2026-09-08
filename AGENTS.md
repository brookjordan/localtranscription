# AGENTS.md — localtranscription

Read this before changing the repository. This project has a deliberate split
between *creation* and *publishing*.

## Creation: this repository is the source of truth

The canonical project, evaluation code, results, and publishable site live in:

```text
~/git/brookjordan/localtranscription/
├── scripts/       transcription runners, corpus tools, checks, banner generation
├── docs/          evaluation notes and technical documentation
├── results/       transcripts, JSON measurements, logs that are safe to track,
│                  and the comparison dashboard
├── site/          canonical project page + atomic devlog + banner images
└── site-assets/   generated banner source files used by site/
```

When creating or revising a project page or devlog post, work in `site/` here:

- `site/index.html` is the project landing page.
- `site/blog/index.html` is the post index.
- `site/blog/YYYY-MM-DD-HHMM-slug.html` is one atomic dated post.
- `site/img/` contains the banners used by the posts.
- Keep posts evidence-led: use dates from the repository, Obsidian project
  tracking, Hermes sessions, and file timestamps. Do not invent chronology.
- Match the localimagegen devlog tone: concise engineering entries, clear
  decisions and failures, dry humour where it is earned, and actual snippets,
  timings, transcripts, or links rather than generic project prose.
- Prefer several small posts over one retrospective essay when the source
  record contains separate experiments or decisions.

The project repository is the canonical authoring location. Do not author a new
post directly in `brookjordan.github.io` and then forget to bring it back here.

## Publishing: copy the canonical site to the website repository

The website repository is a deployment target, not the authoring source:

```text
~/git/brookjordan/brookjordan.github.io/projects/localtranscription/
```

To publish a site update:

1. Finish and verify the change in `localtranscription/site/`.
2. Copy the complete `site/` directory to
   `brookjordan.github.io/projects/localtranscription/`.
3. Inspect the diff and ensure unrelated website changes are untouched.
4. Commit and push the website repository separately.
5. Verify the published project URL and every referenced public asset.

A normal copy command is:

```zsh
rsync -a --delete localtranscription/site/ \
  brookjordan.github.io/projects/localtranscription/
```

`--delete` is intentional only when the source and destination are confirmed;
never point it at the wrong directory.

## Audio and other large assets

Audio is not committed to either Git repository. Authorised sample files are
served from the NAS asset share at:

```text
https://assets.brook.dev/localtranscription/<exact-file>
```

On Brook's Mac, the real SMB share should be mounted at `/tmp/nas-assets`:

```zsh
mount_smbfs '//Brook%20Jordan@ug-nas-x4800.tail4148b0.ts.net/assets' /tmp/nas-assets
```

Do not trust `/Users/brook.jordan/NAS/assets`; it can be an ordinary local
folder when the SMB share is absent. Copy large files to
`/tmp/nas-assets/localtranscription/`, then verify the exact public URL returns
HTTP 200. Use a new versioned path rather than overwriting immutable media.

Every recording must have appropriate authorisation before publication. Posts
should link exact audio files; directory browsing is disabled.

## Image generation

ComfyUI runs locally at `http://127.0.0.1:8188`. Before downloading any model,
check disk space and preserve at least 20 GB free. Prefer the existing models
and the standalone `scripts/gen-blog-banners.mjs`; do not install a new model
just to decorate a post. QA generated images visually before committing them.

## Git rules

- Use conventional commits.
- Keep model caches, virtual environments, `vendor/` builds, and private source
  audio out of Git.
- Verify `git status` before committing so unrelated work is not included.
- Push this repository and the website repository as separate operations.
