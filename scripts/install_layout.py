#!/usr/bin/env python3
"""Install the organized source tree into an existing checkout, without Git writes."""
import sys
from pathlib import Path
for directory in (Path(__file__).resolve().parents[1]/'src',Path(__file__).resolve().parent):
    sys.path.insert(0,str(directory))
import argparse
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from common import ROOT,sha256
from package_submission import files
from validate_submission import validate


def install(target, dry_run=False, keep_report=False):
    target=Path(target).expanduser().resolve()
    if target==ROOT:
        raise ValueError('Extract the new package separately; target must be the existing checkout')
    validate(ROOT)
    migration=json.loads((ROOT/'artifacts/layout_migration.json').read_text())
    legacy=migration['legacy_files_sha256']
    accepted=migration.get('accepted_legacy_files_sha256',{})
    payload={p.relative_to(ROOT).as_posix():p for p in files()}
    overrides={}
    report=target/'report.md'
    if keep_report and report.is_file():
        text=report.read_text(encoding='utf-8')
        if not text.strip() or len(text.split())>1500:
            raise ValueError('Existing report must be nonempty and at most 1500 words; no files changed')
        payload.pop('report.md',None)
        name='artifacts/completion_manifest.json'
        manifest=json.loads(payload[name].read_text())
        manifest['files_sha256']['report.md']=sha256(report)
        manifest['retained_report']={'sha256':sha256(report),'source':'Existing checkout; preserved with --keep-report'}
        overrides[name]=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    changes=[];remove=[];conflicts=[]
    for name,source in payload.items():
        existing=target/name
        expected=hashlib.sha256(overrides[name]).hexdigest() if name in overrides else sha256(source)
        if existing.exists():
            if not existing.is_file():
                conflicts.append(name);continue
            actual=sha256(existing)
            if actual==expected:
                continue
            if actual!=legacy.get(name) and actual not in accepted.get(name,[]):
                conflicts.append(name);continue
        changes.append(name)
    for name,expected in legacy.items():
        existing=target/name
        if keep_report and name=='report.md':
            continue
        if name not in payload and existing.exists():
            if not existing.is_file() or sha256(existing)!=expected:
                conflicts.append(name)
            else:
                remove.append(name)
    if conflicts:
        raise ValueError('Local edits detected; no files changed. Review these paths: '+', '.join(sorted(set(conflicts))))
    if not changes and not remove:
        print('Already up to date.');return None
    print(f'Copy {len(changes)} files; remove {len(remove)} obsolete paths.')
    if dry_run:
        return None
    target.mkdir(parents=True,exist_ok=True)
    backup=target.parent/f'{target.name}_before_layout_{time.time_ns()}.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as archive:
        preserved=['report.md'] if keep_report and report.is_file() else []
        for name in sorted(set(changes+remove+preserved)):
            if (target/name).is_file():archive.write(target/name,name)
    for name in changes:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent,delete=False) as temp:
            temporary=Path(temp.name)
        try:
            if name in overrides:
                temporary.write_bytes(overrides[name])
            else:
                shutil.copyfile(payload[name],temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    for name in remove:(target/name).unlink()
    validate(target)
    print('Backup:',backup)
    print('Repository files updated. Git history and index were not modified.')
    return backup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--keep-report',action='store_true',help='Keep the existing report.md verbatim and update its delivery checksum')
    args=parser.parse_args()
    try:
        install(args.repo,args.dry_run,args.keep_report)
    except ValueError as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()
