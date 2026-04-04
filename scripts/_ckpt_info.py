import torch, os, datetime
from pathlib import Path
p = Path('data/models/aquaforge/aquaforge.pt')
mtime = os.path.getmtime(p)
dt = datetime.datetime.fromtimestamp(mtime)
ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
meta = ckpt.get('meta', {})
print(f"Epoch in checkpoint : {meta.get('epoch', 'unknown')}")
print(f"File last saved     : {dt.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"in_channels         : {meta.get('in_channels', 'not set')}")
