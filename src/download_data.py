"""Download immutable source artifacts; never publish the raw cache."""
from pathlib import Path
import hashlib, json, urllib.request, time
ROOT=Path(__file__).resolve().parents[1]
REV='f5d499befd5cedca7d09ee86d2cbbd210a330798'
ORIGINAL_REV='40b436560ca83a8dba36114c22ab3c66e43f6d5e'

def download(url, dest):
    if not dest.exists():
        temp=dest.with_suffix('.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temp.open('wb') as f:
                    while block:=response.read(1024*1024): f.write(block)
                temp.replace(dest)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    return {'url':url,'file':str(dest.relative_to(ROOT)), 'bytes':dest.stat().st_size,
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}

if __name__=='__main__':
    raw=ROOT/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for name in ['0000.parquet','0001.parquet']:
        url=f'https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/{REV}/default/train/{name}'
        print('Downloading',name,flush=True)
        artifact=download(url,raw/name)
        expected={'0000.parquet':'1d3061d5ea00f64c4bec9e53ac899752e01a89ebffa271c461516f5bf95e81ff',
                  '0001.parquet':'e3ee9e436b4c9bd5a8811d02ae3b5a6dff69605cb66719ac47fad61c65e562fe'}
        if artifact['sha256']!=expected[name]:raise ValueError('Source checksum mismatch: '+name)
        artifacts.append(artifact)
        print(artifacts[-1],flush=True)
    card=download(f'https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/{ORIGINAL_REV}/README.md',raw/'dataset_card.md')
    metadata={'dataset':'openbmb/UltraFeedback','original_revision':ORIGINAL_REV,'parquet_revision':REV,
              'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'artifacts':artifacts,'dataset_card':card,
              'license_label':'MIT (dataset card); upstream source terms also apply',
              'notes':'No quality annotations or system prompts are used as predictive inputs. Raw records are kept local.'}
    (ROOT/'docs/data_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
