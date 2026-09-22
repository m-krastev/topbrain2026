"""Extract application source (not the venv) from a `docker save` tarball."""
import io, json, sys, tarfile
from pathlib import Path
image, out, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
keep = ('.py', '.sh', '.md', '.toml', '.cfg', '.txt', '.yaml', '.yml', '.json')
with tarfile.open(image) as outer:
    names = outer.getnames()
    manifest = json.load(outer.extractfile('manifest.json'))
    for layer in manifest[0]['Layers']:
        raw = outer.extractfile(layer).read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as inner:
            for m in inner.getmembers():
                n = m.name.lstrip('./')
                if not m.isfile() or not n.startswith(prefix) or n.startswith(('usr/','lib','etc/','bin/','sbin/','var/','opt/conda','root/.cache','tmp/')) or '/.venv/' in n or '/site-packages/' in n:
                    continue
                if not n.endswith(keep) or m.size > 5_000_000:
                    continue
                dest = out / n
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(inner.extractfile(m).read())
print('done')
