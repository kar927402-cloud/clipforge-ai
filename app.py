"""ClipForge AI free backend. Uses local faster-whisper + FFmpeg; no paid AI API."""
import gc, os, json, uuid, subprocess, threading, shutil, zipfile, re, math, datetime, importlib.util
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, abort

DATA = Path(os.getenv("DATA_DIR", "data")).resolve(); DATA.mkdir(exist_ok=True)
app = Flask(__name__, static_folder="static", static_url_path="/static")
# Allow the Android WebView / hosted frontend to call this backend.
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response
MAXMB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
app.config["MAX_CONTENT_LENGTH"] = MAXMB * 1024 * 1024
LOCK = threading.RLock()
SEM = threading.Semaphore(int(os.getenv("RENDER_WORKERS", "1")))
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
ALLOW_YT = False  # URL importing is intentionally disabled; upload videos you own or have permission to edit.
EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
LANGS = {"en": "english", "hi": "hindi"}
FONTS = ["Noto Sans", "DejaVu Sans", "Noto Serif", "Liberation Sans"]
# No OpenAI or other paid API client is used.

# ---------- storage ----------
def pfile(pid):
    if not re.fullmatch(r"[a-f0-9]{12}", pid): abort(404)
    return DATA / pid / "project.json"

def load(pid):
    f = pfile(pid)
    if not f.exists(): abort(404)
    with LOCK: return json.loads(f.read_text())

def mutate(pid, fn):
    with LOCK:
        p = load(pid); fn(p)
        f = pfile(pid); t = f.with_suffix(".tmp"); t.write_text(json.dumps(p)); t.replace(f)
        return p

def brief(p): return {k: v for k, v in p.items() if k != "segments"}
def src_of(pid):
    g = list((DATA / pid).glob("source.*")); return g[0] if g else None

def sh(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode: raise RuntimeError((r.stderr or "command failed")[-500:])
    return r.stdout

def duration(path):
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())

# ---------- local pipeline: transcribe -> heuristic highlights ----------
def words(text):
    return re.findall(r"[\w’'-]+", (text or "").lower(), flags=re.UNICODE)

HOOK_WORDS = {"why", "how", "secret", "truth", "mistake", "never", "always", "important", "because", "लेकिन", "क्यों", "कैसे", "सच", "गलती", "जरूरी", "कभी", "इसलिए", "मतलब"}

def choose_highlights(segs, total, clip_len, count):
    # Build candidate windows from transcript timestamps, then rank by speech density,
    # hook vocabulary, numbers and sentence boundaries. This is local heuristic ranking,
    # not a generative LLM and does not promise virality.
    candidates = []
    step = max(4, clip_len * 0.35)
    starts = [max(0.0, min(max(0.0, total-clip_len), float(g["s"]))) for g in segs]
    starts += [max(0.0, total-clip_len)]
    seen = set()
    for raw in starts:
        a = max(0.0, min(max(0.0, total-clip_len), round(raw / step) * step))
        key = round(a, 1)
        if key in seen: continue
        seen.add(key); b = min(total, a + clip_len)
        part = [g for g in segs if g["e"] > a and g["s"] < b]
        if not part: continue
        txt = " ".join(g["t"] for g in part).strip()
        ws = words(txt)
        if len(ws) < 5: continue
        hooks = sum(1 for w in ws if w in HOOK_WORDS)
        numbers = sum(1 for w in ws if any(ch.isdigit() for ch in w))
        punctuation = sum(txt.count(x) for x in "?!।.")
        density = min(1.0, len(ws) / max(1, clip_len) / 2.2)
        score = int(max(1, min(99, 35 + density*30 + min(hooks,5)*5 + min(numbers,3)*3 + min(punctuation,5)*2)))
        title = " ".join(txt.split()[:9]).strip()[:90] or "Interesting moment"
        first = re.split(r"(?<=[.!?।])\s+", txt)[0][:180]
        tags = []
        for w in ws:
            if len(w) >= 5 and w not in tags:
                tags.append(w)
            if len(tags) == 3: break
        hashtags = ["#" + re.sub(r"[^\w]", "", t) for t in tags if re.sub(r"[^\w]", "", t)]
        candidates.append({"title": title, "hook": first or title, "caption": txt[:1000], "hashtags": hashtags,
                           "score": score, "reason": "Local transcript-based highlight estimate", "start": round(a,1),
                           "end": round(b,1), "focus": 0.5, "render": None})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for c in candidates:
        if all(c["end"] <= x["start"] or c["start"] >= x["end"] for x in selected):
            selected.append(c)
            if len(selected) >= count: break
    if not selected and segs:
        g = segs[0]; a = max(0.0, min(g["s"], max(0.0,total-clip_len)))
        b = min(total, a+clip_len); txt = " ".join(x["t"] for x in segs if x["e"]>a and x["s"]<b)
        if txt.strip(): selected.append({"title": "Video highlight", "hook": txt[:160], "caption": txt[:1000], "hashtags": [],
            "score": 50, "reason": "First available speech segment", "start": round(a,1), "end": round(b,1), "focus": .5, "render": None})
    for i,c in enumerate(selected): c["id"] = f"c{i+1}"
    return selected

def process(pid):
    def st(stage, prog): mutate(pid, lambda p: p.update(status="processing", stage=stage, progress=prog, error=None))
    try:
        p = load(pid); d = DATA / pid
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("FFmpeg and ffprobe must be installed on the backend server.")
        src = src_of(pid)
        if not src: raise RuntimeError("Video file is missing. Upload an MP4/MOV/MKV/WEBM file.")
        total = duration(src)
        if total <= 0: raise RuntimeError("Could not read video duration.")
        mutate(pid, lambda q: q.update(duration=total))
        st("Loading local speech model", 12)
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("Local transcription package is missing. Install requirements.txt first.")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        st("Transcribing with local Whisper model", 35)
        seg_iter, info = model.transcribe(str(src), beam_size=1, vad_filter=True)
        segs = []
        for x in seg_iter:
            t = (x.text or "").strip()
            if t: segs.append({"s": float(x.start), "e": float(x.end), "t": t})
        del model; gc.collect()  # free RAM before rendering
        if not segs: raise RuntimeError("No speech detected. Try a video with clear spoken audio.")
        lang = (getattr(info, "language", "") or "").lower()
        mutate(pid, lambda q: q.update(segments=segs, lang=lang))
        st("Finding highlights locally", 75)
        p = load(pid)
        clips = choose_highlights(segs, total, p["clip_len"], p["clip_count"])
        if not clips: raise RuntimeError("Could not find enough clear speech for clips. Try a longer video.")
        mutate(pid, lambda q: q.update(clips=clips, status="ready", stage="Ready", progress=100))
    except Exception as ex:
        mutate(pid, lambda q: q.update(status="error", stage="Failed", error=str(ex)[:500]))

# ---------- rendering ----------
def hexc(c):
    c = (c or "#FFFFFF").lstrip("#"); return f"&H00{c[4:6]}{c[2:4]}{c[0:2]}".upper()
def tsa(t): return f"{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}"

def build_cues(segs, a, b):
    out = []
    for g in segs:
        if g["e"] <= a or g["s"] >= b: continue
        s, e = max(g["s"], a) - a, min(g["e"], b) - a
        w = g["t"].split()
        if not w: continue
        k = math.ceil(len(w) / 5)
        for i in range(k): out.append((s + (e - s) * i / k, s + (e - s) * (i + 1) / k, " ".join(w[i*5:(i+1)*5])))
    return out


def render_clip(pid, cid):
    def setr(**kw):
        def fn(p):
            for c in p["clips"]:
                if c["id"] == cid: c["render"] = {**(c.get("render") or {}), **kw}
        mutate(pid, fn)
    try:
        with SEM:
            setr(status="rendering", error=None)
            p = load(pid); c = next(x for x in p["clips"] if x["id"] == cid); st = p["style"]
            W, H = (1080, 1920) if st["res"] == "1080" else (720, 1280)
            cues = build_cues(p["segments"], c["start"], c["end"])
            tgt = st["lang"]
            # Captions remain in the original spoken language; no paid translation API is used.
            fs = int(H * 0.042)
            ass = ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n[V4+ Styles]\n"
                   "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
                   "Style: D,%s,%d,%s,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,%d,1,2,60,60,%d,1\n\n"
                   "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
                   % (W, H, st["font"], fs, hexc(st["color"]), max(2, fs // 12), int(H * 0.12)))
            for a, b, t in cues:
                ass += f"Dialogue: 0,{tsa(a)},{tsa(b)},D,,0,0,0,,{t.replace('{', '(').replace('}', ')')}\n"
            cd = DATA / pid / "clips"; cd.mkdir(exist_ok=True)
            (cd / f"{cid}.ass").write_text(ass, encoding="utf-8")
            f = max(0.0, min(1.0, float(c.get("focus", .5))))
            vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}:(iw-{W})*{f}:(ih-{H})/2,ass={cid}.ass"
            sh(["ffmpeg", "-y", "-ss", str(c["start"]), "-i", str(src_of(pid).resolve()), "-t", str(c["end"] - c["start"]),
                "-threads", "2", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", f"{cid}.mp4"], cwd=cd)
            setr(status="done", file=f"{cid}.mp4", res=st["res"], size=(cd / f"{cid}.mp4").stat().st_size)
    except Exception as ex:
        setr(status="error", error=str(ex)[-300:])

# ---------- API ----------
def err(m, code=400): return jsonify(error=m), code

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/status")
def status():
    return jsonify(ai=bool(importlib.util.find_spec("faster_whisper")), local_ai=True, ffmpeg=bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), youtube=ALLOW_YT, max_upload_mb=MAXMB, fonts=FONTS, model=MODEL_SIZE)

@app.get("/api/projects")
def plist():
    out = []
    for f in DATA.glob("*/project.json"):
        try: out.append(brief(json.loads(f.read_text())))
        except Exception: pass
    out.sort(key=lambda p: p["created"], reverse=True)
    return jsonify(out)

@app.post("/api/projects")
def create():
    url = (request.form.get("url") or "").strip(); f = request.files.get("file")
    try: dur = int(request.form.get("clip_len", 30)); cnt = int(request.form.get("clip_count", 5))
    except ValueError: return err("Invalid options")
    if dur not in (15, 30, 45, 60) or not 1 <= cnt <= 15: return err("Invalid clip length or count")
    pid = uuid.uuid4().hex[:12]; d = DATA / pid
    if f and f.filename:
        ext = Path(f.filename).suffix.lower()
        if ext not in EXTS: return err("Unsupported file type. Use MP4, MOV, MKV, WEBM, AVI or M4V.")
        d.mkdir(); f.save(d / f"source{ext}"); title, kind = Path(f.filename).stem, "upload"
    elif url:
        return err("YouTube URL importing is disabled in the free build. Upload a video you own or have permission to edit.")
        if not ALLOW_YT: return err("YouTube import is disabled on this server.")
        if not re.match(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", url): return err("That is not a YouTube URL.")
        if request.form.get("authorized") != "true": return err("Confirm you own this video or have permission to use it.")
        d.mkdir(); title, kind = url, "youtube"
    else: return err("Provide a video file or a YouTube URL.")
    p = {"id": pid, "title": title, "kind": kind, "url": url if kind == "youtube" else None, "clip_len": dur, "clip_count": cnt,
         "created": datetime.datetime.utcnow().isoformat() + "Z", "status": "processing", "stage": "Queued", "progress": 1,
         "error": None, "clips": [], "segments": [], "duration": 0, "lang": "",
         "style": {"lang": "auto", "font": FONTS[0], "color": "#FFFFFF", "res": "720"}}
    (d / "project.json").write_text(json.dumps(p))
    threading.Thread(target=process, args=(pid,), daemon=True).start()
    return jsonify(id=pid)

@app.get("/api/projects/<pid>")
def get(pid): return jsonify(load(pid))

@app.delete("/api/projects/<pid>")
def delete(pid):
    load(pid); shutil.rmtree(DATA / pid, ignore_errors=True); return jsonify(ok=True)

@app.post("/api/projects/<pid>/retry")
def retry(pid):
    if load(pid)["status"] != "error": return err("Nothing to retry.")
    mutate(pid, lambda q: q.update(status="processing", stage="Retrying", progress=1, error=None))
    threading.Thread(target=process, args=(pid,), daemon=True).start()
    return jsonify(ok=True)

@app.get("/api/projects/<pid>/source")
def source(pid):
    load(pid); s = src_of(pid)
    return send_file(s, conditional=True) if s else abort(404)

@app.put("/api/projects/<pid>/edits")
def edits(pid):
    j = request.get_json(force=True, silent=True) or {}
    def fn(p):
        for e in j.get("clips", []):
            c = next((x for x in p["clips"] if x["id"] == e.get("id")), None)
            if not c: continue
            old = (c["start"], c["end"], c["focus"])
            try:
                s, en = float(e.get("start", c["start"])), float(e.get("end", c["end"]))
                if 0 <= s < en <= p["duration"] + 0.05: c["start"], c["end"] = round(s, 1), round(en, 1)
                c["focus"] = max(0.0, min(1.0, float(e.get("focus", c["focus"]))))
            except (TypeError, ValueError): pass
            c["title"] = str(e.get("title", c["title"]))[:100]; c["caption"] = str(e.get("caption", c["caption"]))[:1000]
            if old != (c["start"], c["end"], c["focus"]): c["render"] = None
        s = j.get("style") or {}
        new = {"lang": s.get("lang") if s.get("lang") in ("auto", "en", "hi") else p["style"]["lang"],
               "font": s.get("font") if s.get("font") in FONTS else p["style"]["font"],
               "color": s.get("color") if re.fullmatch(r"#[0-9a-fA-F]{6}", str(s.get("color", ""))) else p["style"]["color"],
               "res": s.get("res") if s.get("res") in ("720", "1080") else p["style"]["res"]}
        if new != p["style"]:
            p["style"] = new
            for c in p["clips"]: c["render"] = None
    return jsonify(brief(mutate(pid, fn)))

@app.post("/api/projects/<pid>/render")
def render(pid):
    p = load(pid)
    if p["status"] != "ready": return err("Project is not ready.")
    ids = [i for i in ((request.get_json(silent=True) or {}).get("ids") or [c["id"] for c in p["clips"]]) if any(c["id"] == i for c in p["clips"])]
    def fn(q):
        for c in q["clips"]:
            if c["id"] in ids: c["render"] = {"status": "queued"}
    mutate(pid, fn)
    for i in ids: threading.Thread(target=render_clip, args=(pid, i), daemon=True).start()
    return jsonify(ok=True)

@app.get("/api/projects/<pid>/clips/<cid>/download")
def dl(pid, cid):
    c = next((x for x in load(pid)["clips"] if x["id"] == cid), None); r = c and c.get("render")
    if not r or r.get("status") != "done": return err("Clip is not rendered yet.", 404)
    name = re.sub(r"[^\w\- ]+", "", c["title"]).strip()[:60] or cid
    return send_file(DATA / pid / "clips" / r["file"], mimetype="video/mp4", as_attachment=True, download_name=f"{name}.mp4")

@app.get("/api/projects/<pid>/zip")
def zipall(pid):
    p = load(pid); z = DATA / pid / "clips.zip"; n = 0
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for i, c in enumerate(p["clips"], 1):
            r = c.get("render")
            if r and r.get("status") == "done": zf.write(DATA / pid / "clips" / r["file"], f"{i:02d}-{c['id']}.mp4"); n += 1
    return send_file(z, as_attachment=True, download_name="clipforge-clips.zip") if n else err("No rendered clips yet.", 404)

def recover():  # jobs interrupted by a restart
    for f in DATA.glob("*/project.json"):
        try:
            p = json.loads(f.read_text()); ch = False
            if p["status"] == "processing": p.update(status="error", stage="Failed", error="Server restarted during processing. Press Retry."); ch = True
            for c in p["clips"]:
                if c.get("render") and c["render"].get("status") in ("queued", "rendering"): c["render"] = None; ch = True
            if ch: f.write_text(json.dumps(p))
        except Exception: pass

recover()  # also runs under gunicorn, so stuck jobs reset after a restart

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), threaded=True)
