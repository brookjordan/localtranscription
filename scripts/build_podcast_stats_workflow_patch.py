#!/usr/bin/env python3
"""Build the n8n update payload for the podcast stats sidecar UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_detail")
    parser.add_argument("output")
    args = parser.parse_args()

    outer = json.loads(Path(args.workflow_detail).read_text(encoding="utf-8"))
    workflow = json.loads(outer["result"])["workflow"]
    node = next(item for item in workflow["nodes"] if item["name"] == "Build podcast webpage")
    source = node["parameters"]["jsCode"]
    updated = source.replace(
        "const assetBaseUrl = publication.assetBaseUrl;\n",
        "const assetBaseUrl = publication.assetBaseUrl;\n"
        "const publicStats = $('Build public stats JSON').first().json;\n"
        "const publicStatsJson = JSON.stringify(publicStats, null, 2);\n",
    )
    updated = updated.replace(
        "// Grab all incoming items directly from the HTTP Request node\nconst items = $input.all();",
        "// Preserve the complete media collection even though stats collection is now inline.\n"
        "const items = $('Record assets written').all();",
    )
    css = """
        .stats-panel { margin-top: 1rem; text-align: left; border: 1px solid #3a3a3a; border-radius: 14px; background: #202020; }
        .stats-panel summary { cursor: pointer; padding: 12px 14px; color: #bb86fc; font-weight: 700; list-style-position: inside; }
        .stats-content { padding: 0 14px 14px; color: #ddd; }
        .stats-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 8px 0 12px; }
        .stats-card { background: #292929; border-radius: 10px; padding: 10px; }
        .stats-value { display: block; color: #fff; font-size: 1.05rem; font-weight: 750; }
        .stats-label { display: block; color: #999; font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em; margin-top: 3px; }
        .stats-stage { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; border-top: 1px solid #333; padding: 9px 0; font-size: .8rem; }
        .stats-stage small { grid-column: 1 / -1; color: #888; }
        .stats-raw { color: #bb86fc; font-size: .78rem; }
        .stats-error { color: #ffb4ab; }
"""
    updated = updated.replace("    </style>\n</head>", css + "    </style>\n</head>")
    markup = """
        <details id="stats-panel" class="stats-panel">
            <summary>Stats for nerds</summary>
            <div id="stats-content" class="stats-content" aria-live="polite">Open to load run statistics.</div>
        </details>
"""
    updated = updated.replace(
        '        <button id="toggle-chat" class="toggle-btn">Show Full Transcript</button>',
        markup + '\n        <button id="toggle-chat" class="toggle-btn">Show Full Transcript</button>',
    )
    script = """
        const statsPanel = document.getElementById('stats-panel');
        const statsContent = document.getElementById('stats-content');
        let statsLoaded = false;
        const formatDuration = (ms) => ms == null ? 'Unavailable' : ms < 1000 ? ms + ' ms' : (ms / 1000).toFixed(ms < 60000 ? 1 : 0) + ' s';
        const formatBytes = (bytes) => bytes == null ? 'Unavailable' : (bytes / (1024 ** 3)).toFixed(1) + ' GiB';
        function renderStats(stats) {
            const summary = stats.summary || {};
            const speech = stats.speech || {};
            const cards = [
                ['Total run', formatDuration(summary.wallClockMs)],
                ['Peak memory', formatBytes(summary.peakHostUsedBytes)],
                ['Speech attempts', String(speech.totalAttempts ?? 'Unavailable')],
                ['Speech retries', String(speech.retries ?? 'Unavailable')],
            ];
            const stageRows = (stats.stages || []).map((stage) =>
                '<div class="stats-stage"><strong>' + stage.stage + '</strong><span>' + formatDuration(stage.durationMs) + '</span>' +
                '<small>' + stage.status + '</small></div>'
            ).join('');
            statsContent.innerHTML = '<div class="stats-grid">' + cards.map(([label, value]) =>
                '<div class="stats-card"><span class="stats-value">' + value + '</span><span class="stats-label">' + label + '</span></div>'
            ).join('') + '</div>' + stageRows + '<a class="stats-raw" href="stats.json" target="_blank" rel="noopener">View raw stats.json</a>';
        }
        statsPanel.addEventListener('toggle', async () => {
            if (!statsPanel.open || statsLoaded) return;
            statsLoaded = true;
            statsContent.textContent = 'Loading run statistics…';
            try {
                const response = await fetch('stats.json', { cache: 'no-store' });
                if (!response.ok) throw new Error('HTTP ' + response.status);
                renderStats(await response.json());
            } catch (error) {
                statsContent.innerHTML = '<p class="stats-error">Statistics are unavailable for this episode.</p>';
            }
        });
"""
    updated = updated.replace(
        "        const toggleBtn = document.getElementById('toggle-chat');\n",
        "        const toggleBtn = document.getElementById('toggle-chat');\n" + script,
    )
    updated = updated.replace(
        "additions.push({\n  path: `projects/news/menu-data.json`,\n  contents: Buffer.from(menuDataJson).toString('base64'),\n});",
        "additions.push({\n  path: `projects/news/menu-data.json`,\n  contents: Buffer.from(menuDataJson).toString('base64'),\n});\n"
        "additions.push({\n  path: `projects/news/${folderName}/stats.json`,\n"
        "  contents: Buffer.from(publicStatsJson).toString('base64'),\n});",
    )
    if updated == source:
        raise SystemExit("No replacements applied")
    for marker in ("Build public stats JSON", "stats-panel", "stats.json", "Record assets written"):
        if marker not in updated:
            raise SystemExit(f"Missing marker: {marker}")

    payload = {
        "workflowId": workflow["id"],
        "operations": [
            {
                "type": "updateNodeParameters",
                "nodeName": "Build podcast webpage",
                "parameters": {"jsCode": updated},
                "replace": True,
            },
            {
                "type": "setNodeGroups",
                "nodeGroups": [{
                    "id": "ef8ac460-a138-4eff-af13-66c71a06548e",
                    "name": "Publish and verify podcast",
                    "nodeNames": [
                        "Prepare publication directory", "Create NAS episode directory",
                        "Prepare final media manifest", "Download final media",
                        "Normalise final media", "Write final media to NAS",
                        "Record assets written", "Record publication stats checkpoint",
                        "Build public stats JSON", "Build podcast webpage",
                        "Record publication inputs", "Fetch publishing branch head",
                        "Commit podcast webpage", "Check GitHub commit result",
                        "Verify published page on GitHub", "Assert publication verified",
                    ],
                    "description": "Persist assets and stats, commit the episode page, then verify the exact page from master before reporting success.",
                }],
            },
        ],
        "versionName": "Add on-demand stats viewer",
        "versionDescription": "Publishes stats.json with each episode and adds an accessible, lazy-loaded Stats for nerds panel plus a raw JSON link. Restores the publication node group with the new stats nodes included.",
        "skillsUsed": [
            "using-n8n-skills-official", "n8n-workflow-lifecycle-official",
            "n8n-node-configuration-official", "n8n-expressions-official",
            "n8n-code-nodes-official",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {output} ({len(updated)} JS characters)")


if __name__ == "__main__":
    main()
