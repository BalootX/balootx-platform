import {createHash} from 'node:crypto';
import {mkdtemp, readdir, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, relative, resolve, sep} from 'node:path';
import {createClient} from '@hey-api/openapi-ts';
import configuredClient from '../openapi-ts.config.mjs';

const userConfig = await configuredClient;
if (Array.isArray(userConfig) || typeof userConfig.output === 'string') {
  throw new Error('The BalootX OpenAPI generator requires one object output configuration.');
}

const trackedOutput = resolve(userConfig.output.path);
const temporaryRoot = await mkdtemp(join(tmpdir(), 'balootx-openapi-'));
const temporaryOutput = join(temporaryRoot, 'generated');

try {
  await createClient({
    ...userConfig,
    logs: {file: false, level: 'silent'},
    output: {...userConfig.output, path: temporaryOutput},
  });

  const [trackedFiles, temporaryFiles] = await Promise.all([
    listFiles(trackedOutput),
    listFiles(temporaryOutput),
  ]);
  const allFiles = [...new Set([...trackedFiles, ...temporaryFiles])].sort();
  const failures = [];

  for (const file of allFiles) {
    const tracked = trackedFiles.includes(file)
      ? await readFile(join(trackedOutput, file))
      : null;
    const generated = temporaryFiles.includes(file)
      ? await readFile(join(temporaryOutput, file))
      : null;
    if (tracked === null) {
      failures.push(`missing tracked file: ${file}`);
    } else if (generated === null) {
      failures.push(`stale tracked file: ${file}`);
    } else if (!tracked.equals(generated)) {
      failures.push(
        `changed file: ${file} (${digest(tracked)} != ${digest(generated)})`,
      );
    }
  }

  if (failures.length > 0) {
    throw new Error(
      `Generated OpenAPI client drift detected:\n- ${failures.join('\n- ')}\n` +
        'Run `npm run generate:api` and review the complete generated change.',
    );
  }

  process.stdout.write(
    `Generated OpenAPI client is current (${trackedFiles.length} files).\n`,
  );
} finally {
  const safePrefix = `${resolve(tmpdir())}${sep}balootx-openapi-`;
  if (!resolve(temporaryRoot).startsWith(safePrefix)) {
    throw new Error(`Refusing to remove unexpected temporary path: ${temporaryRoot}`);
  }
  await rm(temporaryRoot, {recursive: true, force: true});
}

async function listFiles(root) {
  const files = [];
  await walk(root, files);
  return files.sort();

  async function walk(directory, output) {
    let entries;
    try {
      entries = await readdir(directory, {withFileTypes: true});
    } catch (error) {
      if (error?.code === 'ENOENT') return;
      throw error;
    }
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await walk(path, output);
      } else if (entry.isFile()) {
        output.push(relative(root, path).split(sep).join('/'));
      }
    }
  }
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex').slice(0, 12);
}
