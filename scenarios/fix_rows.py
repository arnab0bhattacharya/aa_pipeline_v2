"""Replace premise/setting for given ids in a drafts file. Usage: fix_rows.py AUTHOR fixes.json"""
import json, sys
author, fixes = sys.argv[1], json.load(open(sys.argv[2], encoding='utf-8'))
p = f'drafts/{author}.jsonl'
rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
for r in rows:
    if r['id'] in fixes:
        r['premise'], r['setting'] = fixes[r['id']]
open(p, 'w', encoding='utf-8').write('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
print(author, len(rows), 'rows;', len(fixes), 'replaced')
