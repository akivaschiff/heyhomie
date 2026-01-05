import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';

const execAsync = promisify(exec);

// Use ncc to bundle
await execAsync('npx ncc build build/index.js -o build/dist');

// Move the bundled file to replace the original
fs.renameSync('build/dist/index.js', 'build/index.js');
fs.rmSync('build/dist', { recursive: true });

console.log('✓ Bundle created');
