import {defineConfig} from '@hey-api/openapi-ts';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const dashboardRoot = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  input: resolve(dashboardRoot, '../../../../api/openapi.json'),
  interactive: false,
  logs: {file: false, level: 'warn'},
  output: {
    path: resolve(dashboardRoot, 'src/api/generated'),
    clean: true,
    entryFile: true,
    module: {extension: null},
    postProcess: [],
    preferExportAll: true,
    source: false,
    tsConfigPath: resolve(dashboardRoot, 'tsconfig.json'),
  },
  plugins: ['@hey-api/client-fetch', '@hey-api/typescript', '@hey-api/sdk'],
});
