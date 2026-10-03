// Canonical packaging fallback for preserved figure frames. The packaged
// headless checker times out on these frames; do not label this a browser pass.
// Usage: node analysis/geometry_v1_package.mjs PLUGIN_ROOT REPORT_DIRECTORY
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [pluginRoot, reportDirectory] = process.argv.slice(2);
if (!pluginRoot || !reportDirectory) throw Error('Expected PLUGIN_ROOT REPORT_DIRECTORY');
const out = path.resolve(reportDirectory);
const script = name => pathToFileURL(path.join(pluginRoot, 'skills/build-report/scripts', name)).href;
const {buildPortableArtifact} = await import(script('build_portable_artifact.mjs'));
const {verifyPortableArtifactStructure} = await import(script('verify_portable_artifact.mjs'));
const current = JSON.parse(fs.readFileSync(path.join(out, 'artifact.json'), 'utf8'));
const previous = JSON.parse(fs.readFileSync(path.join(out, 'before_diagrams/artifact.json'), 'utf8'));
if (JSON.stringify(current.manifest.charts) !== JSON.stringify(previous.manifest.charts) ||
    JSON.stringify(current.snapshot.datasets) !== JSON.stringify(previous.snapshot.datasets)) {
  throw Error('Native chart definitions or data changed; regenerate their static vectors through report:deliver.');
}
const staticCharts = JSON.parse(fs.readFileSync(path.join(out, 'native_chart_vectors.json'), 'utf8'));
if (Object.keys(staticCharts).length !== 3) throw Error('Missing native chart vectors');
const candidate = path.join(out, 'report.candidate.html');
fs.writeFileSync(candidate, buildPortableArtifact(current, {staticCharts}));
const result = verifyPortableArtifactStructure({artifactPath:path.join(out, 'artifact.json'), htmlPath:candidate});
if (!result.ok) throw Error('Structural verification failed');
fs.renameSync(candidate, path.join(out, 'report.html'));
console.log(JSON.stringify({validation:'passed', package:'passed', structural_verification:'passed',
  browser_verification:'manual review required; headless figure-frame startup timeout', output:path.join(out, 'report.html')}));
