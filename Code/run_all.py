"""Run the nine numerical verification and experiment scripts."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'Data').mkdir(exist_ok=True)
(ROOT/'Reports/logs').mkdir(parents=True,exist_ok=True)
for script in sorted((ROOT/'Code/scripts').glob('s*.py')):
    result=subprocess.run([sys.executable,str(script)],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'Reports/logs'/f'{script.stem}.txt').write_text(result.stdout+result.stderr)
    print(('PASS' if result.returncode==0 else 'FAIL'),script.name,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
