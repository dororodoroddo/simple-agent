import { spawnSync, spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const pythonVersion = readFileSync(path.join(root, '.python-version'), 'utf8').trim();
export const pythonEnv = {
  ...process.env,
  UV_PYTHON_INSTALL_DIR: path.join(root, '.local/python'),
  UV_MANAGED_PYTHON: '1',
  PATH: [path.join(root, '.venv/bin'), path.join(root, '.local/tooling'), process.env.PATH].join(path.delimiter),
};

function uvCommand() {
  for (const file of ['.local/tooling/uv', '.venv/bin/uv']) {
    if (existsSync(path.join(root, file))) return [path.join(root, file), []];
  }
  return ['uv', ['tool', 'run', '--from', 'uv==0.13.5', 'uv']];
}

export function runUv(args, options = {}) {
  const [command, prefix] = uvCommand();
  return spawn(command, [...prefix, ...args], { cwd: root, env: pythonEnv, stdio: 'inherit', ...options });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const [command, prefix] = uvCommand();
  const commands = args[0] === 'setup'
    ? [['python', 'install', '--install-dir', pythonEnv.UV_PYTHON_INSTALL_DIR, '--no-bin', pythonVersion], ['sync', '--locked', '--all-groups']]
    : [args];
  for (const argv of commands) {
    const result = spawnSync(command, [...prefix, ...argv], { cwd: root, env: pythonEnv, stdio: 'inherit' });
    if (result.error) throw result.error;
    if (result.status !== 0) process.exit(result.status ?? 1);
  }
}
