"""CrimeVision AI - Intelligent Multimodal Forensic Analysis Platform
Skill Wallet Capstone Project 2. Single-file, no external DB. Run: streamlit run app.py
Detections/transcripts in DEMO mode are SIMULATED. All AI output requires human review."""
import io, json, re, hashlib, random, base64, tempfile, os, datetime as dt
from xml.sax.saxutils import escape
import requests, pandas as pd, streamlit as st
from PIL import Image, ImageDraw

try:
    import cv2, numpy as np
except Exception:
    cv2 = None

st.set_page_config(page_title="CrimeVision AI", page_icon="🕵️", layout="wide")
S = st.session_state
for k, v in dict(ledger=[], evidence={}, img=None, aud=None, vid=None, chat=[], provider="Groq", key="").items():
    S.setdefault(k, v)

st.markdown("""<style>
.stApp{background:#0b0f14;color:#d7e0ea}
h1,h2,h3{color:#00e5ff;letter-spacing:.5px}
section[data-testid=stSidebar]{background:#10161d;border-right:1px solid #1e2a36}
.badge{display:inline-block;padding:3px 10px;border-radius:4px;font:600 12px monospace;margin:2px 4px 2px 0;border:1px solid}
.ok{color:#00e676;border-color:#00e676}.warn{color:#ffab00;border-color:#ffab00}.bad{color:#ff1744;border-color:#ff1744}
.card{background:#111922;border:1px solid #1e2a36;border-left:4px solid #00e5ff;padding:12px 16px;border-radius:6px;margin-bottom:8px}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers
def utcnow(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def fmt(s): s = int(s); return f"T+{s//60:02d}:{s%60:02d}"
def make_id(prefix, data): return f"{prefix}-{hashlib.sha256(data).hexdigest()[:8].upper()}"
def seeded(data): return random.Random(int(hashlib.md5(data).hexdigest(), 16))
def secret(name):
    try: return st.secrets.get(name, "")
    except Exception: return ""

def register(eid, kind, name, data):
    S.evidence[eid] = dict(id=eid, type=kind, file=name, sha256=hashlib.sha256(data).hexdigest(),
                           logged=utcnow(), custodian=S.get("inv") or "UNASSIGNED")

def add_events(eid, source, events):
    S.ledger = [e for e in S.ledger if e["evidence"] != eid]
    for n, e in enumerate(events, 1):
        S.ledger.append(dict(id=f"{eid}-{n:02d}", source=source, evidence=eid, t=float(e["t"]), time=fmt(e["t"]),
                             desc=e["desc"], category=e.get("category", "Observation"), conf=round(float(e.get("conf", 1)), 2)))
    S.ledger.sort(key=lambda e: e["t"])

# ---------------------------------------------------------------- LLM layer
def llm(prompt, media=None):
    p, k = S.provider, S.key
    system = "You are a forensic analysis assistant. Be precise. Never invent facts."
    if p == "Groq":
        if media:
            b64 = base64.b64encode(media[0]).decode()
            content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{media[1]};base64,{b64}"}}]
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            content, model = prompt, "llama-3.3-70b-versatile"
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k}"},
                          json=dict(model=model, temperature=0.1, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}]), timeout=90)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    parts = [{"text": system + "\n\n" + prompt}]
    if media: parts.append({"inline_data": {"mime_type": media[1], "data": base64.b64encode(media[0]).decode()}})
    r = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                      headers={"x-goog-api-key": k}, json=dict(contents=[{"parts": parts}], generationConfig=dict(temperature=0.1)), timeout=120)
    r.raise_for_status(); return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def ask_json(prompt, media=None):
    """Returns parsed JSON or None (caller falls back to demo mode)."""
    if not S.key: return None
    try:
        txt = llm(prompt + "\nReturn ONLY valid JSON, no markdown.", media)
        m = re.search(r"[\[{].*[\]}]", txt, re.S)
        return json.loads(m.group(0))
    except Exception as e:
        st.warning(f"Live API call failed ({type(e).__name__}); using demo fallback.")
        return None

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🕵️ CrimeVision AI")
    S.provider = st.selectbox("LLM provider", ["Groq", "Gemini"])
    S.key = st.text_input("API key (blank = demo mode)", type="password",
                          value=secret("GROQ_API_KEY" if S.provider == "Groq" else "GEMINI_API_KEY"))
    live = bool(S.key)
    st.markdown(f'<span class="badge {"ok" if live else "warn"}">{"LIVE · " + S.provider if live else "DEMO MODE (simulated)"}</span>', unsafe_allow_html=True)
    st.divider(); st.markdown("### Chain of Custody")
    case_id = st.text_input("Case ID", "CASE-2026-0001")
    S["inv"] = st.text_input("Investigator name / badge", key="inv_in")
    signed = st.checkbox("I certify evidence was handled per protocol (sign-off)")
    sign_ts = utcnow() if signed else "—"
    st.markdown(f'<span class="badge {"ok" if S["inv"] else "bad"}">INVESTIGATOR: {"SET" if S["inv"] else "MISSING"}</span>'
                f'<span class="badge {"ok" if signed else "warn"}">SIGN-OFF: {"YES" if signed else "PENDING"}</span>', unsafe_allow_html=True)
    st.caption(f"Session: {utcnow()}"); st.caption(f"Evidence items logged: {len(S.evidence)}")
    st.divider(); st.caption("⚠️ AI output is investigative support, not evidence of guilt. Human review required.")

st.title("CrimeVision AI")
st.caption("Intelligent Multimodal Forensic Analysis Platform · Skill Wallet Capstone Project 2")
tabs = st.tabs(["🖼️ Scene Tagger", "🎙️ Witness Audio", "🎥 CCTV Analysis", "🧬 Fusion & Evidence Chat", "📄 PDF Report"])

# ---------------------------------------------------------------- Tab 1: image
CLASSES = {"Person": "#00e5ff", "Firearm": "#ff1744", "Edged weapon": "#ff9100", "Blood spatter": "#d50000", "Forced entry point": "#ffea00"}
LEADS = {"Firearm": "Secure firearm; request ballistics, serial trace and latent prints.",
         "Edged weapon": "Bag blade for DNA/fingerprints; match against wound patterns.",
         "Blood spatter": "Photograph with scale; swab for DNA; run bloodstain pattern analysis.",
         "Forced entry point": "Collect tool-mark impressions; check neighbouring cameras for ingress route.",
         "Person": "Identify all persons present; record statements and clothing/appearance."}

def detect(data, mime):
    if live:
        j = ask_json('Act as an RT-DETR style object detector for crime-scene photos. Detect ONLY visible instances of: '
                     + ", ".join(CLASSES) + '. Return {"detections":[{"label":..., "confidence":0-1, "bbox":[x,y,w,h] normalized 0-1}]}. Empty list if none.', (data, mime))
        if j is not None:
            out = []
            for d in j.get("detections", []):
                try:
                    x, y, w, h = [min(max(float(v), 0), 1) for v in d["bbox"]]
                    if d["label"] in CLASSES: out.append(dict(label=d["label"], conf=round(float(d["confidence"]), 2), bbox=[x, y, w, h]))
                except Exception: continue
            return out
    r = seeded(data); out = []
    for lab in r.sample(list(CLASSES), r.randint(3, 5)):
        for _ in range(2 if lab == "Person" else 1):
            w, h = r.uniform(.1, .25), r.uniform(.12, .3)
            out.append(dict(label=lab, conf=round(r.uniform(.62, .97), 2), bbox=[r.uniform(0, 1 - w), r.uniform(0, 1 - h), w, h]))
    return out

def assess(dets):
    L = {d["label"] for d in dets}; f, e, b, n = "Firearm" in L, "Edged weapon" in L, "Blood spatter" in L, "Forced entry point" in L
    lvl = "Critical" if b and (f or e) else "High" if f or e else "Medium" if b or n else "Low"
    cls = "Homicide" if b and (f or e) else "Assault" if f or e or b else "Burglary" if n else "Vandalism"
    return lvl, cls

with tabs[0]:
    st.subheader("Crime Scene Visual Evidence Tagger")
    up = st.file_uploader("Upload scene photograph", type=["jpg", "jpeg", "png"], key="img_up")
    off = st.number_input("Capture time offset (seconds from incident T0)", 0, 86400, 0, key="img_off")
    if up and st.button("🔍 Analyze scene", type="primary"):
        try:
            data = up.getvalue(); im = Image.open(io.BytesIO(data)).convert("RGB"); im.thumbnail((1280, 1280))
            dets = detect(data, up.type or "image/jpeg"); eid = make_id("IMG", data); register(eid, "Image", up.name, data)
            ann = im.copy(); dr = ImageDraw.Draw(ann); W, H = ann.size
            for d in dets:
                x, y, w, h = d["bbox"]; box = [x * W, y * H, (x + w) * W, (y + h) * H]; c = CLASSES[d["label"]]
                dr.rectangle(box, outline=c, width=3); dr.text((box[0] + 3, max(box[1] - 12, 0)), f'{d["label"]} {d["conf"]:.2f}', fill=c)
            lvl, cls = assess(dets); buf = io.BytesIO(); ann.save(buf, "JPEG")
            S.img = dict(eid=eid, dets=dets, ann=buf.getvalue(), lvl=lvl, cls=cls, size=(W, H), mode="LIVE" if live else "DEMO")
            add_events(eid, "Image", [dict(t=off, desc=f'{d["label"]} detected at x={d["bbox"][0]:.2f}, y={d["bbox"][1]:.2f}',
                                          category=d["label"], conf=d["conf"]) for d in dets])
        except Exception as ex: st.error(f"Image analysis failed: {ex}")
    r = S.img
    if r:
        c1, c2 = st.columns([3, 2]); c1.image(r["ann"], caption=f'Evidence {r["eid"]} · {r["mode"]}', use_container_width=True)
        col = {"Low": "ok", "Medium": "warn", "High": "bad", "Critical": "bad"}[r["lvl"]]
        c2.markdown(f'<div class="card"><b>Threat Level</b><br><span class="badge {col}">{r["lvl"].upper()}</span><br><br><b>Scene Classification</b><br>{r["cls"]}</div>', unsafe_allow_html=True)
        W, H = r["size"]
        c2.dataframe(pd.DataFrame([dict(Object=d["label"], Confidence=d["conf"], X=int(d["bbox"][0] * W), Y=int(d["bbox"][1] * H),
                                        W=int(d["bbox"][2] * W), H=int(d["bbox"][3] * H)) for d in r["dets"]]), hide_index=True, use_container_width=True)
        st.markdown("**Prioritized investigative leads**")
        order = ["Firearm", "Edged weapon", "Blood spatter", "Forced entry point", "Person"]
        for i, lab in enumerate([l for l in order if any(d["label"] == l for d in r["dets"])], 1): st.markdown(f"{i}. **{lab}** — {LEADS[lab]}")
        if not r["dets"]: st.info("No target objects detected.")

# ---------------------------------------------------------------- Tab 2: audio
KEYS = {"Weapon mention": ["gun", "pistol", "knife", "weapon", "armed"], "Flight": ["ran", "fled", "running", "drove off", "took off"],
        "Forced entry": ["window", "broke", "break-in", "glass", "kicked"], "Injury": ["blood", "bleeding", "hurt", "shot", "stabbed"],
        "Vehicle": ["car", "sedan", "van", "truck", "bike"]}
DEMO_AUDIO = [(0, "Dispatcher", "112, what is your emergency?"), (4, "Caller", "I heard glass break and someone broke the back window next door."),
              (11, "Dispatcher", "Can you see anyone?"), (15, "Caller", "A man with a gun is inside. There's blood on the floor."),
              (24, "Caller", "He just ran out the back door!"), (29, "Caller", "He got into a dark sedan and drove off fast.")]

def transcribe(data, name, mime):
    if live:
        try:
            if S.provider == "Groq":
                r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers={"Authorization": f"Bearer {S.key}"},
                                  files={"file": (name, data)}, data={"model": "whisper-large-v3", "response_format": "verbose_json"}, timeout=180)
                r.raise_for_status(); segs = [(s["start"], s["text"].strip()) for s in r.json().get("segments", [])]
                tag = ask_json('Assign speaker tags (e.g. Caller, Dispatcher, Witness 1). Return list of {"t":seconds,"speaker":..,"text":..}. Input: ' + json.dumps(segs))
                return tag or [dict(t=t, speaker="Speaker", text=x) for t, x in segs]
            j = ask_json('Transcribe this audio. Return list of {"t":seconds,"speaker":..,"text":..}.', (data, mime or "audio/mpeg"))
            if j: return j
        except Exception as ex: st.warning(f"Transcription failed ({ex}); using demo transcript.")
    return [dict(t=t, speaker=s, text=x) for t, s, x in DEMO_AUDIO]

with tabs[1]:
    st.subheader("Witness Audio Transcription & Timeline")
    up = st.file_uploader("Upload witness statement / 911-112 call", type=["wav", "mp3", "m4a", "ogg", "flac"], key="aud_up")
    off = st.number_input("Audio start offset (seconds from incident T0)", 0, 86400, 0, key="aud_off")
    if up and st.button("📝 Transcribe & analyze", type="primary"):
        try:
            data = up.getvalue(); eid = make_id("AUD", data); register(eid, "Audio", up.name, data)
            segs = transcribe(data, up.name, up.type); evs = []
            for s in segs:
                low = s["text"].lower(); hit = [k for k, w in KEYS.items() if any(re.search(rf"\b{x}\b", low) for x in w)]
                evs.append(dict(t=off + float(s["t"]), desc=f'[{s["speaker"]}] {s["text"]}', category=", ".join(hit) or "Statement", conf=1.0 if live else .8))
            S.aud = dict(eid=eid, segs=segs, mode="LIVE" if live else "DEMO"); add_events(eid, "Audio", evs)
        except Exception as ex: st.error(f"Audio analysis failed: {ex}")
    if S.aud:
        st.caption(f'Evidence {S.aud["eid"]} · {S.aud["mode"]}')
        evs = [e for e in S.ledger if e["evidence"] == S.aud["eid"]]
        st.dataframe(pd.DataFrame([dict(Time=e["time"], Marker=e["id"], Event=e["desc"], KeyEvent=e["category"]) for e in evs]), hide_index=True, use_container_width=True)
        st.markdown("**Key events**")
        for e in evs:
            if e["category"] != "Statement": st.markdown(f'- `{e["time"]}` **{e["category"]}** — {e["desc"]} `[{e["id"]}]`')

# ---------------------------------------------------------------- Tab 3: video
def track_video(path, sps=2):
    if cv2 is None: raise RuntimeError("opencv not installed")
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(fps / sps), 1); bg = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=40, detectShadows=False)
    rows, tracks, nid = [], {}, 1
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (320, int(f.shape[0] * 320 / f.shape[1]))); H, W = f.shape[:2]; t = i / fps
        m = cv2.morphologyEx(bg.apply(f), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); used = set()
        for c in cs:
            if cv2.contourArea(c) < 0.004 * W * H: continue
            x, y, w, h = cv2.boundingRect(c); cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lab = "Vehicle" if w / max(h, 1) > 1.4 and w * h > 0.03 * W * H else "Person"
            best = min(((abs(tr["x"] - cx) + abs(tr["y"] - cy), k) for k, tr in tracks.items() if k not in used and t - tr["t"] < 2), default=(9, None))
            if best[0] < 0.25: tid = best[1]
            else: tid = nid; nid += 1
            used.add(tid); tracks[tid] = dict(x=cx, y=cy, t=t); rows.append(dict(track=f"{lab[0]}{tid}", label=lab, t=round(t, 1), x=round(cx, 3), y=round(cy, 3)))
    cap.release(); return pd.DataFrame(rows), n / fps

def demo_tracks():
    rows = [dict(track="P1", label="Person", t=t, x=.1 + .08 * t, y=.6 - .02 * t) for t in range(0, 9)]
    rows += [dict(track="V1", label="Vehicle", t=t, x=.9 - .2 * (t - 8), y=.75) for t in range(8, 13)]
    return pd.DataFrame(rows), 13.0

def video_events(df):
    evs = []
    for tid, g in df.groupby("track"):
        g = g.sort_values("t"); lab = g.label.iloc[0]; t0, t1 = g.t.iloc[0], g.t.iloc[-1]
        evs.append(dict(t=t0, desc=f"{lab} {tid} enters frame at ({g.x.iloc[0]:.2f},{g.y.iloc[0]:.2f})",
                        category="Criminal trespass / unlawful presence (suspected)" if lab == "Person" else "Vehicle presence", conf=.7))
        if len(g) > 1:
            dist = ((g.x.diff() ** 2 + g.y.diff() ** 2) ** .5).sum(); sp = dist / max(t1 - t0, .5)
            if sp > .08: evs.append(dict(t=t0 + (t1 - t0) / 2, desc=f"{lab} {tid} rapid movement (speed {sp:.2f} frame-widths/s)",
                                         category="Flight from scene (suspected)" if lab == "Person" else "Getaway vehicle movement (suspected)", conf=.65))
        evs.append(dict(t=t1, desc=f"{lab} {tid} exits frame at ({g.x.iloc[-1]:.2f},{g.y.iloc[-1]:.2f})", category="Departure", conf=.7))
    ppl = df[df.label == "Person"]
    for t, g in ppl.groupby("t"):
        if g.track.nunique() > 1 and ((g.x.max() - g.x.min()) ** 2 + (g.y.max() - g.y.min()) ** 2) ** .5 < .12:
            evs.append(dict(t=t, desc="Two persons in close proximity", category="Assault / physical altercation (suspected)", conf=.55)); break
    return evs

with tabs[2]:
    st.subheader("CCTV Video Analysis & Event Progression")
    up = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    off = st.number_input("Video start offset (seconds from incident T0)", 0, 86400, 0, key="vid_off")
    if up and st.button("🎞️ Process video", type="primary"):
        try:
            data = up.getvalue(); eid = make_id("VID", data); register(eid, "Video", up.name, data); mode = "REAL (OpenCV motion tracking)"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tf: tf.write(data)
                df, dur = track_video(tf.name); os.unlink(tf.name)
            except Exception as ex:
                st.warning(f"Video processing unavailable ({ex}); using simulated tracks."); df, dur = demo_tracks(); mode = "SIMULATED"
            evs = video_events(df) if len(df) else []
            for e in evs: e["t"] += off
            S.vid = dict(eid=eid, df=df, dur=dur, mode=mode); add_events(eid, "Video", evs)
        except Exception as ex: st.error(f"Video analysis failed: {ex}")
    v = S.vid
    if v:
        st.caption(f'Evidence {v["eid"]} · {v["mode"]} · duration {v["dur"]:.1f}s · {len(v["df"])} sampled detections')
        if len(v["df"]):
            st.markdown("**Suspect / vehicle trajectories (normalized frame coordinates)**"); st.scatter_chart(v["df"], x="x", y="y", color="track")
        else: st.info("No significant motion detected.")
        st.markdown("**Chronological event log**")
        st.dataframe(pd.DataFrame([dict(Time=e["time"], ID=e["id"], Event=e["desc"], PenalCategory=e["category"], Conf=e["conf"])
                                   for e in S.ledger if e["evidence"] == v["eid"]]), hide_index=True, use_container_width=True)
        st.caption("Penal categories are indicative labels only; legal classification must be confirmed by counsel.")

# ---------------------------------------------------------------- Tab 4: fusion + RAG
CITE = re.compile(r"(?:IMG|AUD|VID)-[0-9A-F]{8}-\d{2}")
STOP = set("a an the was were is are of to in on at before after did do does any and or it he she they his her that this with by for from".split())
def toks(s): return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}
SYN = {"weapon": {"gun", "firearm", "knife", "edged", "weapon"}, "fled": {"fled", "flight", "ran", "exits", "rapid", "drove", "departure"},
       "suspect": {"person", "man"}, "car": {"vehicle", "sedan"}, "blood": {"blood", "spatter"}}
def expand(q): return q | {s for w in q for k, v in SYN.items() if w == k or w in v for s in v | {k}}

def retrieve(q, k=8):
    qt = expand(toks(q)); sc = [(len(qt & toks(e["desc"] + " " + e["category"] + " " + e["source"])), e) for e in S.ledger]
    return [e for s, e in sorted(sc, key=lambda x: (-x[0], x[1]["t"])) if s > 0][:k]

def cite(e): return f'[{e["id"]}]'
def answer(q):
    if not S.ledger: return "The evidence ledger is empty. Analyze evidence in Tabs 1-3 first."
    ql = q.lower(); wp = [e for e in S.ledger if re.search(r"firearm|gun|knife|edged|weapon", e["desc"].lower() + e["category"].lower())]
    fl = [e for e in S.ledger if re.search(r"flight|rapid|ran |fled|drove|exits", e["desc"].lower() + e["category"].lower())]
    if "before" in ql and ("flee" in ql or "fled" in ql or "flight" in ql) and ("weapon" in ql or "gun" in ql):
        if not wp: return "Insufficient evidence: no weapon is recorded in the ledger."
        if not fl: return f"A weapon is recorded {cite(wp[0])} at {wp[0]['time']}, but no flight event is recorded, so the order cannot be established."
        a, b = wp[0], fl[0]
        return (f"Yes. A weapon was first recorded at {a['time']} {cite(a)}, before the earliest flight event at {b['time']} {cite(b)}." if a["t"] < b["t"]
                else f"Not established. The earliest weapon record ({a['time']}) {cite(a)} is not earlier than the first flight event ({b['time']}) {cite(b)}. Note image timestamps are investigator-supplied offsets.")
    hits = retrieve(q)
    if not hits: return "Insufficient evidence: nothing in the ledger matches this question."
    if live:
        try:
            ctx = "\n".join(f'{e["id"]} | {e["time"]} | {e["source"]} | {e["desc"]} | {e["category"]}' for e in (S.ledger if len(S.ledger) <= 120 else hits))
            out = llm("Answer ONLY from the EVIDENCE below. Cite the evidence ID in square brackets after every claim. If the evidence does not answer, reply exactly: Insufficient evidence in ledger.\n\nEVIDENCE:\n" + ctx + f"\n\nQUESTION: {q}")
            ids = set(CITE.findall(out)); valid = {e["id"] for e in S.ledger}
            if ids <= valid and (ids or "Insufficient evidence" in out): return out
        except Exception as ex: st.warning(f"Live chat failed ({ex}); using extractive answer.")
    return "Relevant evidence (extractive, no inference):\n\n" + "\n".join(f'- {e["time"]} · {e["source"]} — {e["desc"]} {cite(e)}' for e in sorted(hits, key=lambda e: e["t"]))

with tabs[3]:
    st.subheader("Master Forensic Event Ledger")
    if S.ledger:
        df = pd.DataFrame(S.ledger)[["time", "id", "source", "desc", "category", "conf"]].rename(columns=str.title)
        m = st.columns(4); m[0].metric("Events", len(df)); [m[i + 1].metric(s, int((df.Source == s).sum())) for i, s in enumerate(["Image", "Audio", "Video"])]
        st.dataframe(df, hide_index=True, use_container_width=True)
        if st.button("🗑️ Clear ledger & chat"): S.ledger, S.chat = [], []; st.rerun()
    else: st.info("Ledger empty — analyze evidence in the previous tabs.")
    st.subheader("Grounded Evidence Chat")
    st.caption("Answers use only ledger entries and cite evidence IDs. Try: “Was a weapon visible before the suspect fled?”")
    for role, msg in S.chat: st.chat_message(role).markdown(msg)
    if q := st.chat_input("Interrogate the evidence…"):
        S.chat.append(("user", q)); a = answer(q); S.chat.append(("assistant", a)); st.rerun()

# ---------------------------------------------------------------- Tab 5: PDF
def build_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLI
    ss = getSampleStyleSheet(); sm = ParagraphStyle("sm", parent=ss["BodyText"], fontSize=7.5, leading=9)
    P = lambda t: Paragraph(escape(str(t)), sm)
    def tbl(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2a3a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 7.5)])); return t
    ledger_hash = hashlib.sha256(json.dumps(S.ledger, sort_keys=True).encode()).hexdigest()
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=18 * mm,
                                                title=f"Forensic Summary {case_id}", author=S["inv"] or "CrimeVision AI")
    st_ = [Paragraph("FORENSIC SUMMARY REPORT", ss["Title"]), Paragraph("CrimeVision AI - Multimodal Forensic Analysis Platform", ss["Normal"]), Spacer(1, 8),
           tbl([["Case ID", case_id], ["Report generated", utcnow()], ["Investigator", S["inv"]], ["Sign-off timestamp", sign_ts],
                ["Analysis mode", "LIVE - " + S.provider if live else "DEMO (simulated)"], ["Ledger SHA-256", ledger_hash]], [40 * mm, 140 * mm]), Spacer(1, 10),
           Paragraph("1. Chain of Custody - Evidence Registry", ss["Heading2"]),
           tbl([["Evidence ID", "Type", "File", "SHA-256", "Logged", "Custodian"]] + [[P(e["id"]), P(e["type"]), P(e["file"]), P(e["sha256"]), P(e["logged"]), P(e["custodian"])] for e in S.evidence.values()],
               [22 * mm, 13 * mm, 28 * mm, 60 * mm, 30 * mm, 27 * mm])]
    if S.img:
        r = S.img; st_ += [Paragraph("2. Visual Scene Analysis", ss["Heading2"]), Paragraph(f'Evidence {r["eid"]} | Threat level: <b>{r["lvl"]}</b> | Scene classification: <b>{r["cls"]}</b>', ss["Normal"]),
                           Spacer(1, 4), RLI(io.BytesIO(r["ann"]), width=110 * mm, height=110 * mm * r["size"][1] / r["size"][0]), Spacer(1, 4),
                           tbl([["Object", "Confidence", "BBox (norm x,y,w,h)"]] + [[d["label"], d["conf"], ", ".join(f"{v:.2f}" for v in d["bbox"])] for d in r["dets"]], [50 * mm, 30 * mm, 80 * mm])]
    st_ += [Paragraph("3. Master Forensic Event Ledger", ss["Heading2"]),
            tbl([["Time", "Evidence ID", "Source", "Event", "Category", "Conf"]] + [[e["time"], P(e["id"]), e["source"], P(e["desc"]), P(e["category"]), e["conf"]] for e in S.ledger],
                [16 * mm, 33 * mm, 14 * mm, 70 * mm, 38 * mm, 9 * mm]), Spacer(1, 10),
            Paragraph("4. Limitations & Disclaimer", ss["Heading2"]),
            Paragraph("Findings were produced by automated AI/computer-vision tools and, in demo mode, simulated data. They are investigative aids, must be independently verified, and do not constitute a determination of guilt. Legal categories are indicative only.", ss["Normal"]),
            Spacer(1, 24), tbl([["Investigator signature", "Date / Time"], ["\n\n" + (S["inv"] or "________________"), sign_ts]], [90 * mm, 90 * mm])]
    foot = lambda c, d: (c.setFont("Helvetica", 7), c.drawString(15 * mm, 10 * mm, f"{case_id} | CONFIDENTIAL - FORENSIC | Page {d.page}"))
    doc.build(st_, onFirstPage=foot, onLaterPages=foot); return buf.getvalue()

with tabs[4]:
    st.subheader("Court-Admissible Forensic PDF Export")
    ready = bool(S.ledger) and bool(S["inv"]) and signed
    for ok, msg in [(bool(S.ledger), "Evidence ledger populated"), (bool(S["inv"]), "Investigator identified"), (signed, "Sign-off completed")]:
        st.markdown(f'<span class="badge {"ok" if ok else "bad"}">{"✔" if ok else "✖"} {msg}</span>', unsafe_allow_html=True)
    if st.button("📄 Generate report", type="primary", disabled=not ready):
        try: S.pdf = build_pdf()
        except Exception as ex: st.error(f"PDF generation failed: {ex}")
    if S.get("pdf") and ready:
        st.download_button("⬇️ Download PDF", S.pdf, file_name=f"{case_id}_forensic_report.pdf", mime="application/pdf")
