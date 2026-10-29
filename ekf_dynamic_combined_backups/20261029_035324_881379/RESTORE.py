#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil


def main():
    backup=Path(__file__).resolve().parent
    info=json.loads((backup/'restore.json').read_text())
    workspace=Path(info['workspace'])
    for rel,item in info['files'].items():
        p=workspace/rel
        current=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if current!=item['new_sha256']:
            raise SystemExit('File changed after installation; nothing restored: '+str(p))
    for rel,item in info['files'].items():
        target=workspace/rel
        if item['existed']:shutil.copy2(backup/rel,target)
        else:target.unlink()
    print('Restored pre-combined-patch files: '+str(workspace))


if __name__=='__main__':main()
