import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = join(SCRIPT_DIR, '..', 'src');
const TEXT_EXTENSIONS = new Set(['.css', '.js', '.jsx', '.json', '.md']);
const FORBIDDEN_PATTERNS = [
  [/鈥|闁|璺|閼|�/, 'mojibake replacement text'],
  [/\.\.\.\{/, 'broken template interpolation'],
];

const walk = (dir) => {
  const files = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    const stats = statSync(path);
    if (stats.isDirectory()) {
      files.push(...walk(path));
    } else if (TEXT_EXTENSIONS.has(extname(path))) {
      files.push(path);
    }
  }
  return files;
};

const failures = [];
for (const file of walk(ROOT)) {
  const text = readFileSync(file, 'utf8');
  for (const [pattern, reason] of FORBIDDEN_PATTERNS) {
    if (pattern.test(text)) {
      failures.push(`${relative(process.cwd(), file)}: ${reason}`);
    }
  }
}

if (failures.length > 0) {
  console.error('Text hygiene check failed:');
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log('Text hygiene check passed.');