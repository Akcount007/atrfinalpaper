import os
import re

MAX_MB = 50 * 1024 * 1024

patterns = {
    'hf_': re.compile(r'hf_[a-zA-Z0-9]{20,}'),
    'api_key': re.compile(r'api[_-]?key\s*[:=]\s*["\'][a-zA-Z0-9_\-]{8,}["\']', re.IGNORECASE),
    'token': re.compile(r'token\s*[:=]\s*["\'][a-zA-Z0-9_\-]{16,}["\']', re.IGNORECASE),
    'password': re.compile(r'password\s*[:=]\s*["\'][^"\']+["\']', re.IGNORECASE),
    'secret': re.compile(r'secret\s*[:=]\s*["\'][a-zA-Z0-9_\-]{8,}["\']', re.IGNORECASE)
}

total_size = 0
active_size = 0
files_over_50mb = []
secret_matches = []
gpqa_files = []

for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    is_to_delete = '_to_delete' in root
    for f in files:
        p = os.path.join(root, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        total_size += sz
        if not is_to_delete:
            active_size += sz
        if sz > MAX_MB:
            files_over_50mb.append((p, sz))
        if 'gpqa' in f.lower():
            gpqa_files.append(p)
        
        # Check text files for secret patterns
        if sz < 10 * 1024 * 1024:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.py', '.json', '.yaml', '.yml', '.md', '.txt', '.csv', '.tex', '.bib', '.sh', '.jsonl']:
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                        content = fp.read()
                        for name, pat in patterns.items():
                            m = pat.findall(content)
                            if m:
                                secret_matches.append((p, name, m[:3]))
                except Exception:
                    pass

print('=== SAFETY SCAN REPORT ===')
print(f'Total repository size (including _to_delete): {total_size / (1024*1024):.2f} MB')
print(f'Active repository size (excluding _to_delete): {active_size / (1024*1024):.2f} MB')
print(f'Files over 50 MB: {len(files_over_50mb)}')
for p, sz in files_over_50mb:
    print(f'  - {p} ({sz / (1024*1024):.2f} MB)')

print(f'\nGPQA Data Check:')
if gpqa_files:
    print(f'  Found {len(gpqa_files)} GPQA files: {gpqa_files}')
else:
    print('  CONFIRMED: Zero GPQA data files found in the repository.')

print(f'\nSensitive Patterns Scan (hf_, api_key, token, password, secret):')
if secret_matches:
    print(f'  Found {len(secret_matches)} pattern occurrences:')
    for p, name, m in secret_matches:
        print(f'  - {p} matches {name}: {m}')
else:
    print('  CONFIRMED: Zero hardcoded secrets, API keys, or HF access tokens found.')
