import io, json
import numpy as np
from PIL import Image
from app.config import MODEL_DIR

# Architecture definitions mirror the training notebook exactly.
def cbr(i, o):
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.f=nn.Sequential(cbr(i,o),cbr(o,o))
    def forward(self,x): return self.f(x)

class ResBlock(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.f=nn.Sequential(cbr(i,o),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o))
        self.sk=nn.Identity() if i==o else nn.Sequential(nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o))
    def forward(self,x): return F.relu(self.f(x)+self.sk(x))

class AttentionGate(nn.Module):
    def __init__(self,ch_g,ch_x,ch_int):
        super().__init__()
        self.wg=nn.Sequential(nn.Conv2d(ch_g,ch_int,1),nn.BatchNorm2d(ch_int))
        self.wx=nn.Sequential(nn.Conv2d(ch_x,ch_int,1),nn.BatchNorm2d(ch_int))
        self.psi=nn.Sequential(nn.Conv2d(ch_int,1,1),nn.BatchNorm2d(1),nn.Sigmoid())
    def forward(self,g,x): return x*self.psi(F.relu(self.wg(g)+self.wx(x)))

def build_unet_family(f=16, block="conv", attention=False):
    import torch.nn as nn
    import torch.nn.functional as F
    Blk = ConvBlock if block=="conv" else ResBlock
    ch=[f,f*2,f*4,f*8,f*16]
    class UNetFamily(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention=attention
            self.enc=nn.ModuleList([Blk(3,ch[0])]+[Blk(ch[i-1],ch[i]) for i in range(1,4)])
            self.pool=nn.MaxPool2d(2)
            self.bott=Blk(ch[3],ch[4])
            self.up=nn.ModuleList([nn.ConvTranspose2d(ch[i+1],ch[i],2,2) for i in (3,2,1,0)])
            self.att=nn.ModuleList([AttentionGate(ch[i],ch[i],max(ch[i]//2,8)) for i in (3,2,1,0)]) if attention else None
            self.dec=nn.ModuleList([Blk(ch[i]*2,ch[i]) for i in (3,2,1,0)])
            self.head=nn.Conv2d(ch[0],1,1)
        def forward(self,x):
            skips=[]
            for e in self.enc:
                x=e(x); skips.append(x); x=self.pool(x)
            x=self.bott(x)
            for k,i in enumerate((3,2,1,0)):
                x=self.up[k](x); s=skips[i]
                if self.attention: s=self.att[k](x,s)
                x=self.dec[k](torch.cat([s,x],1))
            return self.head(x)
    return UNetFamily()

def build_unetpp(f=16, depth=5):
    import torch.nn as nn
    class UNetPP(nn.Module):
        def __init__(self):
            super().__init__()
            ch=[f*2**i for i in range(depth)]; self.depth=depth
            self.pool=nn.MaxPool2d(2)
            self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
            self.blocks=nn.ModuleDict()
            for j in range(depth):
                for i in range(depth-j):
                    cin=(3 if i==0 else ch[i-1]) if j==0 else ch[i]*j+ch[i+1]
                    self.blocks[f"{i}_{j}"]=ConvBlock(cin,ch[i])
            self.head=nn.Conv2d(ch[0],1,1)
        def forward(self,x):
            n={}
            for i in range(self.depth):
                n[(i,0)]=self.blocks[f"{i}_0"](x if i==0 else self.pool(n[(i-1,0)]))
            for j in range(1,self.depth):
                for i in range(self.depth-j):
                    cat=torch.cat([n[(i,k)] for k in range(j)]+[self.up(n[(i+1,j-1)])],1)
                    n[(i,j)]=self.blocks[f"{i}_{j}"](cat)
            return self.head(n[(0,self.depth-1)])
    return UNetPP()

class BrainTumorService:
    def __init__(self):
        self.cfg=json.loads((MODEL_DIR/"brain_tumor"/"deploy_config.json").read_text())
        self.device=None
        self.models={}

    def load_model(self,name):
        import torch
        if name in self.models: return self.models[name]
        self.device=self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f=int(self.cfg["base_channels"])
        builders={
            "UNet":lambda:build_unet_family(f),
            "ResUNet":lambda:build_unet_family(f,block="res"),
            "AttentionUNet":lambda:build_unet_family(f,attention=True),
            "UNet++":lambda:build_unetpp(f)
        }
        if name not in builders: raise ValueError("Unknown model variant")
        model=builders[name]().to(self.device)
        weight=MODEL_DIR/"brain_tumor"/f"{name}.pt"
        if not weight.exists(): raise RuntimeError(f"Missing model weights: {weight}")
        state=torch.load(weight,map_location=self.device,weights_only=True)
        model.load_state_dict(state); model.eval()
        self.models[name]=model
        return model

    def segment(self, raw: bytes, model_name=None, threshold=None):
        import torch
        with Image.open(io.BytesIO(raw)) as im:
            original=im.convert("RGB")
            size=int(self.cfg["img_size"])
            resized=original.resize((size,size),Image.Resampling.BILINEAR)
            arr=np.asarray(resized,dtype=np.float32)/255.0
        mean=np.asarray(self.cfg["mean"],np.float32)
        std=np.asarray(self.cfg["std"],np.float32)
        x=torch.from_numpy(((arr-mean)/std).transpose(2,0,1).copy())[None].to(self.device or torch.device("cpu"))
        name=model_name or self.cfg["model"]
        model=self.load_model(name)
        thr=float(threshold if threshold is not None else self.cfg["threshold"])
        with torch.inference_mode():
            prob=torch.sigmoid(model(x))[0,0].cpu().numpy()
        # Raw prediction before minimum-area filtering
        raw_mask = prob > thr
        raw_pixels = int(raw_mask.sum())

       # Apply minimum-area filtering
        mask = raw_mask.copy()
        min_area = int(self.cfg["min_area"])

        if raw_pixels < min_area:
         mask[:] = False
        # Return mask and probability map as PNG data URLs; nearest-neighbour restores original size.
        
        mask_img=Image.fromarray((mask*255).astype(np.uint8)).resize(original.size,Image.Resampling.NEAREST)
        prob_img=Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).resize(original.size,Image.Resampling.BILINEAR)
        buf=io.BytesIO(); mask_img.save(buf,format="PNG")
        import base64
        mask_b64=base64.b64encode(buf.getvalue()).decode("ascii")
        buf=io.BytesIO(); prob_img.save(buf,format="PNG")
        prob_b64=base64.b64encode(buf.getvalue()).decode("ascii")
        return {
    "model": name,
    "threshold": thr,
    "tumor_pixels_model_resolution": int(mask.sum()),
    "raw_pixels_above_threshold": raw_pixels,
    "probability_min": float(prob.min()),
    "probability_max": float(prob.max()),
    "probability_mean": float(prob.mean()),
    "min_area": min_area,
    "mask_png_base64": mask_b64,
    "probability_map_png_base64": prob_b64,
    "notice": "Research prototype only; not a medical diagnosis."
}

service=BrainTumorService()
