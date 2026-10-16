"""GenAI Studio: FastAPI + Stable Diffusion (text-to-image, img2img, LoRA)."""
import base64, io, os, random, threading, time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw
import uvicorn

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sd-turbo")
LORA_PATH = os.getenv("LORA_PATH")
MOCK = os.getenv("MOCK", "0") == "1"
SIZE = 512
lock = threading.Lock()
t2i = i2i = None
device = "mock"

def load_models():
    global t2i, i2i, device
    if MOCK:
        return
    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    fp16 = device == "cuda"
    print(f"Loading {MODEL_ID} on {device} ...")
    t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if fp16 else torch.float32,
        variant="fp16" if fp16 else None).to(device)
    if LORA_PATH:
        t2i.load_lora_weights(LORA_PATH)
    if not fp16:
        t2i.enable_attention_slicing()           # lower memory on CPU/MPS
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)  # shares weights, no extra VRAM
    t2i.set_progress_bar_config(disable=True)
    i2i.set_progress_bar_config(disable=True)

def mock_image(prompt, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(img)
    c1 = tuple(rnd.randint(20, 235) for _ in range(3))
    c2 = tuple(rnd.randint(20, 235) for _ in range(3))
    for y in range(SIZE):
        t = y / SIZE
        d.line([(0, y), (SIZE, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(c1, c2)))
    d.text((20, SIZE - 30), f"MOCK: {prompt[:60]}", fill="white")
    return img

app = FastAPI(title="GenAI Studio")

@app.get("/api/health")
def health():
    return {"model": MODEL_ID, "device": device, "mock": MOCK, "lora": LORA_PATH}

@app.post("/api/generate")
def generate(prompt: str = Form(...), negative: str = Form(""), steps: int = Form(4),
             guidance: float = Form(0.0), strength: float = Form(0.6), seed: int = Form(-1),
             image: UploadFile | None = File(None)):
    if not prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    init = None
    if image and image.filename:
        init = Image.open(io.BytesIO(image.file.read())).convert("RGB").resize((SIZE, SIZE))
    start = time.time()
    with lock:  # one GPU job at a time
        if MOCK:
            time.sleep(0.8)
            out = mock_image(prompt, seed)
        else:
            import torch
            gen = torch.Generator(device="cpu").manual_seed(seed)
            kw = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen)
            if guidance > 1:
                kw["negative_prompt"] = negative or None
            if init is not None:
                steps_needed = max(steps, int(2 / max(strength, 0.1)) + 1)  # turbo models need steps*strength >= 1
                kw.update(image=init, strength=strength, num_inference_steps=steps_needed)
                out = i2i(**kw).images[0]
            else:
                out = t2i(height=SIZE, width=SIZE, **kw).images[0]
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "seed": seed, "seconds": round(time.time() - start, 2), "device": device,
            "mode": "img2img" if init is not None else "text-to-image"}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenAI Studio</title>
<style>
:root{--paper:#edefe9;--panel:#f8f8f4;--ink:#1c2024;--mute:#68707a;--line:#cfd3cb;--accent:#2d49f0;--r:10px}
@media(prefers-color-scheme:dark){:root{--paper:#14171a;--panel:#1c2024;--ink:#eceee8;--mute:#8f98a3;--line:#30363d;--accent:#7e93ff}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:22px 28px 6px}h1{margin:0;font:600 26px/1.2 Georgia,"Times New Roman",serif;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--mute)}
main{display:grid;grid-template-columns:340px 1fr;gap:22px;padding:18px 28px 40px;max-width:1150px}
@media(max-width:820px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:18px}
label{display:block;font-weight:600;margin:14px 0 5px}label:first-child{margin-top:0}
textarea,input[type=text],input[type=number]{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:6px;background:var(--paper);color:var(--ink);font:inherit}
textarea{min-height:84px;resize:vertical}
input[type=range]{width:100%;accent-color:var(--accent)}
.row{display:flex;justify-content:space-between;align-items:center}.row span{color:var(--mute);font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{border:1px solid var(--line);background:none;color:var(--ink);padding:4px 10px;border-radius:99px;cursor:pointer;font:inherit;font-size:13px}
.chip:hover{border-color:var(--accent)}
button.go{width:100%;margin-top:18px;padding:12px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:600 15px inherit;cursor:pointer}
button.go:disabled{opacity:.6;cursor:wait}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.stage{aspect-ratio:1;max-width:512px;width:100%;border:1px solid var(--line);border-radius:var(--r);background:var(--panel);display:grid;place-items:center;overflow:hidden;color:var(--mute);text-align:center}
.stage img{width:100%;height:100%;object-fit:cover;display:block}
.meta{margin:10px 0;color:var(--mute);font-size:13px;min-height:20px}
.meta a{color:var(--accent)}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}
.strip img{width:84px;height:84px;object-fit:cover;border-radius:6px;border:2px solid transparent;cursor:pointer}
.strip img:hover{border-color:var(--accent)}
.err{color:#c0392b}
.spin{animation:s 1s linear infinite;display:inline-block}@keyframes s{to{transform:rotate(360deg)}}
@media(prefers-reduced-motion:reduce){.spin{animation:none}}
</style></head><body>
<header><h1>GenAI Studio</h1><p>Describe an image, or upload one and restyle it. Runs on a local Stable Diffusion model.</p></header>
<main>
<section class="panel">
 <label for="prompt">Prompt</label>
 <textarea id="prompt">a lighthouse on a cliff at dawn, watercolor, soft light</textarea>
 <div class="chips" id="chips"></div>
 <label for="file">Starting image (optional)</label>
 <input type="file" id="file" accept="image/*">
 <div id="strWrap" hidden><label class="row">Change amount <span id="strV">0.60</span></label>
  <input type="range" id="strength" min="0.2" max="1" step="0.05" value="0.6"></div>
 <label class="row" for="steps">Steps <span id="stepsV">4</span></label>
 <input type="range" id="steps" min="1" max="12" value="4">
 <label for="seed">Seed (-1 = random)</label>
 <input type="number" id="seed" value="-1">
 <button class="go" id="go">Generate image</button>
</section>
<section>
 <div class="stage" id="stage">Your image will appear here.</div>
 <div class="meta" id="meta"></div>
 <div class="strip" id="strip" aria-label="History"></div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
["a fox astronaut floating above Earth, studio lighting","a rainy Kochi street at night, neon reflections, cinematic","isometric tiny village on a leaf, 3D render","portrait of a robot painter, oil on canvas"]
 .forEach(t=>{const b=document.createElement("button");b.className="chip";b.textContent=t.split(",")[0];b.onclick=()=>$("prompt").value=t;$("chips").append(b)});
$("steps").oninput=e=>$("stepsV").textContent=e.target.value;
$("strength").oninput=e=>$("strV").textContent=(+e.target.value).toFixed(2);
$("file").onchange=e=>$("strWrap").hidden=!e.target.files.length;
function show(r){$("stage").innerHTML=`<img src="${r.image}" alt="Generated image">`;
 $("meta").innerHTML=`${r.mode} · seed ${r.seed} · ${r.seconds}s on ${r.device} · <a href="${r.image}" download="genai-${r.seed}.png">Download</a>`;}
$("go").onclick=async()=>{
 const btn=$("go");btn.disabled=true;btn.textContent="Generating…";
 $("stage").innerHTML='<span class="spin">◐</span>';$("meta").textContent="";
 const f=new FormData();f.append("prompt",$("prompt").value);f.append("steps",$("steps").value);
 f.append("seed",$("seed").value);f.append("strength",$("strength").value);
 if($("file").files[0])f.append("image",$("file").files[0]);
 try{const res=await fetch("/api/generate",{method:"POST",body:f});
  if(!res.ok)throw new Error((await res.json()).detail||res.statusText);
  const r=await res.json();show(r);
  const t=document.createElement("img");t.src=r.image;t.alt="Earlier result";t.onclick=()=>show(r);$("strip").prepend(t);
 }catch(e){$("stage").innerHTML="Nothing generated yet.";$("meta").innerHTML=`<span class="err">Failed: ${e.message}</span>`}
 btn.disabled=false;btn.textContent="Generate image";};
</script></body></html>"""

if __name__ == "__main__":
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
