"""Append a read-only FBS comparison to an existing successful Polydesk run."""
import argparse, hashlib, json
from pathlib import Path
from polydesk_compare import compare, write_report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
    folder=Path(__file__).resolve().parent
    pointer=args.project/'runs'/'LATEST'
    if pointer.is_dir(): run=pointer.resolve()
    else: run=Path((args.project/'runs'/'LATEST.txt').read_text(encoding='utf-8-sig').strip())
    if run.parent.resolve() != (args.project/'runs').resolve(): raise ValueError('Unexpected run directory')
    model_file=folder/'app_data.json';model=json.loads(model_file.read_text(encoding='utf-8-sig'))
    result=compare(model,run);result['model_sha256']=hashlib.sha256(model_file.read_bytes()).hexdigest()
    result['research_run']=run.name;out=run/'college-football';write_report(result,out)
    for name in ['STATUS.md','EDGE.md']:
        p=run/name
        note='\n\n## All-FBS model comparison\n\n[Open research comparison](college-football/index.html). '+str(len(result['rows']))+' matched outcomes; research only, execution disabled. The report shows model and market timestamps, venue mismatch and unresolved validation.\n'
        text=p.read_text(encoding='utf-8-sig')
        if '## All-FBS model comparison' not in text:p.write_text(text+note,encoding='utf-8')
    latest=folder/'LATEST.json';temporary=latest.with_suffix('.tmp')
    temporary.write_text(json.dumps({'generated_at':result['generated_at'],'run':run.name,'html':str(out/'index.html'),'rows':len(result['rows']),'execution_enabled':False},indent=2),encoding='utf-8');temporary.replace(latest)
    print(json.dumps({'comparison':str(out/'index.html'),'rows':len(result['rows']),'execution_enabled':False}))
if __name__=='__main__':main()
