"""Interactive 3D replay (self-contained HTML + Three.js).  SEPARATE from training.

Records real episodes from the chosen policies (nothing is simulated by the viewer itself - it only
plays back the JSBSim trajectory) and writes one HTML page you open in a browser.

Examples:
    python tools/replay_3d.py                                   # sac_pilot latest checkpoint (3 seeds) + teacher
    python tools/replay_3d.py --model models/sac_pilot/checkpoint_150000.zip --seeds 1000000 1000001 --level 1
    python tools/replay_3d.py --no-teacher --open

Requires internet in the browser the first time (Three.js is loaded from the unpkg CDN).
Viewer keys: SPACE play/pause, 1-5 cameras (chase, side, tower, orbit, top), +/- speed, R restart.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv  # noqa: E402
from evaluation.evaluate import build_policy  # noqa: E402

DEFAULT_GLB = PROJECT_ROOT / "results" / "replay3d" / "boeing747erf.glb"
FT_PER_M = 3.280839895
# Wheel-contact -> CG height in the FDM when settled (see AIRCRAFT_MODEL_NOTES.md section 9): ~12.5 ft.
CG_HEIGHT_M = 12.49 / FT_PER_M


def frame_row(t: float, st_like: dict) -> list[float]:
    """Pack one pose/instrument frame (compact list; see FRAME_FIELDS in the viewer)."""
    a = st_like["act"]
    return [round(t, 2), round(st_like["along_m"], 2), round(st_like["lateral_m"], 2),
            round(st_like["agl_ft"] / FT_PER_M, 2), round(st_like["roll_deg"], 2), round(st_like["pitch_deg"], 2),
            round(st_like["heading_error_deg"], 2), round(st_like["airspeed_kt"], 1),
            round(st_like["vertical_speed_fpm"], 0), round(float(a[0]), 3), round(float(a[1]), 3),
            round(float(a[2]), 3), round(float(a[3]), 3)]


def record_run(env: Boeing7478TakeoffEnv, policy, label: str, seed: int, level: int) -> dict:
    """Roll out one episode and return a JSON-serialisable run description."""
    obs, info0 = env.reset(seed=seed, options={"level": level})
    s = env._state
    assert s is not None and env.conditions is not None
    frames = [frame_row(0.0, {"along_m": s.along_m, "lateral_m": s.lateral_m, "agl_ft": s.agl_ft, "roll_deg": s.roll_deg,
                              "pitch_deg": s.pitch_deg, "heading_error_deg": s.heading_error_deg,
                              "airspeed_kt": s.airspeed_kt, "vertical_speed_fpm": s.vertical_speed_fpm,
                              "act": np.zeros(4)})]
    while True:
        obs, _r, terminated, truncated, info = env.step(policy(obs))
        info["act"] = info["actuators"]
        frames.append(frame_row(info["time_s"], info))
        if terminated or truncated:
            sm = info["episode_summary"]
            rw = env.conditions.runway
            return {"label": f"{label} - seed {seed}", "reason": sm["reason"], "success": bool(sm["success"]),
                    "duration_s": round(sm["duration_s"], 1), "runway_length_m": round(rw.length_m, 1),
                    "runway_width_m": rw.width_m, "vr_kt": round(info0["vr_est_kt"], 1),
                    "info": (f"{sm['mass_lbs'] / 1000:.0f}k lb | xwind {sm['crosswind_kt']:+.0f} kt, "
                             f"headwind {sm['headwind_kt']:+.0f} kt | mu x{sm['static_friction_factor']:.2f} | "
                             f"elev {sm['runway_elevation_ft']:.0f} ft | gust {sm['gust_sigma_kt']:.1f} kt"),
                    "cg_height_m": round(CG_HEIGHT_M, 2), "frames": frames}


def render_page(runs: list[dict], glb_b64: str = "", glb_cg_from_nose: float = 28.0, live: bool = False) -> str:
    """Return the viewer HTML.  ``live=True`` produces the polling version served by tools/live_viewer.py."""
    page = HTML.replace("__GLB__", glb_b64).replace("__GLB_CG__", repr(float(glb_cg_from_nose)))
    page = page.replace("__LIVE__", "true" if live else "false")
    return page.replace("__RUNS__", json.dumps(runs, separators=(",", ":")))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, nargs="*", default=None, help="SB3 .zip model(s) to replay")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1_000_000, 1_000_001, 1_000_002])
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--no-teacher", action="store_true", help="do not add the teacher run for comparison")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "replay3d" / "replay3d.html"))
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    parser.add_argument("--glb", type=str, default=None,
                        help="detailed aircraft model (.glb); default: results/replay3d/boeing747erf.glb if present")
    parser.add_argument("--glb-cg-from-nose", type=float, default=28.0,
                        help="distance (model metres) from the model's nose to the CG; 28 m for a 747-400-like body")
    args = parser.parse_args()

    configs = load_all_configs()
    models = args.model
    if models is None:
        found = sorted((PROJECT_ROOT / "models" / "sac_pilot").glob("checkpoint_*.zip"),
                       key=lambda p: int(p.stem.split("_")[1]))
        models = [str(found[-1])] if found else []
    env = Boeing7478TakeoffEnv(configs=configs)
    runs: list[dict] = []
    for model_path in models:
        policy = build_policy("teacher", model_path, configs, "cpu")
        label = f"{Path(model_path).parent.name}/{Path(model_path).stem}"
        for seed in args.seeds:
            runs.append(record_run(env, policy, label, seed, args.level))
            print(f"recorded {runs[-1]['label']}: {runs[-1]['reason']} ({runs[-1]['duration_s']} s)")
    if not args.no_teacher:
        teacher = build_policy("teacher", None, configs, "cpu")
        runs.append(record_run(env, teacher, "teacher (PD controller)", args.seeds[0], args.level))
        print(f"recorded {runs[-1]['label']}: {runs[-1]['reason']} ({runs[-1]['duration_s']} s)")
    if not runs:
        raise SystemExit("nothing to replay: pass --model or allow the teacher run")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    glb_b64 = ""
    glb_path = Path(args.glb) if args.glb else DEFAULT_GLB
    if glb_path.is_file():
        glb_b64 = base64.b64encode(glb_path.read_bytes()).decode("ascii")
        print(f"embedding GLB model {glb_path} ({glb_path.stat().st_size / 1e6:.1f} MB)")
    elif args.glb:
        raise SystemExit(f"GLB file not found: {glb_path}")
    out.write_text(render_page(runs, glb_b64, args.glb_cg_from_nose, live=False), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>747-8 research approximation - 3D takeoff replay</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#9cc4e4;font-family:Segoe UI,Arial,sans-serif;color:#fff}
#hud{position:fixed;left:10px;top:10px;background:rgba(10,20,35,.78);padding:7px 10px;border-radius:8px;width:236px;font-size:11.5px}
#hud h3{margin:0 0 3px;font-size:12px;line-height:1.25;word-break:break-word} #cond{font-size:10.5px;line-height:1.3}
.hide-ui #hud,.hide-ui #events,.hide-ui #status,.hide-ui #warn,.hide-ui #credit,.hide-ui #panel{display:none!important}
#hud table{border-collapse:collapse;width:100%} #hud td{padding:1px 4px} #hud td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:7px;background:#345;border-radius:4px;position:relative;margin:2px 0 5px} .bar i{position:absolute;top:0;height:100%;background:#4cf;border-radius:4px}
#panel{position:fixed;left:12px;bottom:12px;right:12px;background:rgba(10,20,35,.78);padding:8px 12px;border-radius:8px;display:flex;gap:12px;align-items:center;font-size:13px}
#panel label{white-space:nowrap} #panel{gap:10px;flex-wrap:nowrap} #panel input[type=range]{flex:1;min-width:80px} select,button{background:#1d3550;color:#fff;border:1px solid #47a;border-radius:5px;padding:4px 8px}
#banner{position:fixed;top:14px;left:50%;transform:translateX(-50%);padding:8px 22px;border-radius:8px;font-size:20px;font-weight:600;display:none}
#warn{position:fixed;right:10px;top:10px;background:rgba(10,20,35,.78);padding:5px 8px;border-radius:8px;font-size:9.5px;max-width:250px;line-height:1.3;opacity:.85}
#events{position:fixed;right:10px;top:84px;width:290px;max-height:30vh;overflow:auto;background:rgba(10,20,35,.78);padding:4px;border-radius:8px;font-size:10.5px;display:none}
#events div{padding:2px 5px;border-radius:4px;cursor:pointer;margin:1px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis} #events div:hover{background:#1d3550} #events div.cur{background:#27517a}
#status{position:fixed;left:10px;bottom:56px;background:rgba(10,20,35,.78);padding:4px 8px;border-radius:8px;font-size:10px;display:none;line-height:1.4;max-width:44vw}
#toast{position:fixed;top:64px;left:50%;transform:translateX(-50%);background:rgba(245,175,0,.96);color:#111;padding:8px 18px;border-radius:8px;font-weight:600;display:none;max-width:70vw;text-align:center}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="hud"><h3 id="title"></h3><div id="cond" style="opacity:.8;margin-bottom:6px"></div><table id="tab"></table>
<div style="margin-top:6px">throttle<div class="bar"><i id="b_thr"></i></div>elevator (+ nose up)<div class="bar"><i id="b_ele"></i></div>
aileron (+ roll right)<div class="bar"><i id="b_ail"></i></div>rudder (+ yaw right)<div class="bar"><i id="b_rud"></i></div></div></div>
<div id="banner"></div>
<div id="credit" style="display:none;position:fixed;right:10px;bottom:56px;background:rgba(10,20,35,.78);padding:3px 8px;border-radius:6px;font-size:9px;max-width:42vw">3D model "Boeing747ERF" by manilov.ap (sketchfab.com/3d-models/boeing747erf-5153fd42231a4723a8634e02c4bb008c), CC-BY-4.0. Scaled to the 747-8 span; visual stand-in only.</div>
<script id="glb" type="text/plain">__GLB__</script>
<div id="events"></div><div id="status"></div><div id="toast"></div>
<div id="warn"><b>Simulation replay.</b> Aircraft: <b>7478_RESEARCH_APPROXIMATION</b> (JSBSim 747-400 FDM with published 747-8 span/area/weight/thrust). Trajectory is real JSBSim output; the 3D model is a simplified visual stand-in. Not certified; simulation only.</div>
<div id="panel"><button id="play">Pause</button><button id="restart">Restart</button>
<label>Run <select id="run"></select></label><label>Camera <select id="cam"><option value="0">1 Chase</option><option value="1">2 Side</option><option value="2">3 Tower</option><option value="3">4 Orbit (drag mouse)</option><option value="4">5 Top-down</option></select></label>
<label>Model <select id="mdl"><option value="glb">Detailed 747 (GLB)</option><option value="simple">Simple</option></select></label>
<label>Speed <select id="spd"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>x</label>
<label id="followlbl" style="display:none"><input type="checkbox" id="follow" checked> Auto-follow newest</label>
<input id="seek" type="range" min="0" max="1000" value="0"><span id="clock">0.0 s</span></div>
<div id="err" style="position:fixed;left:50%;top:60px;transform:translateX(-50%);background:#a00;color:#fff;padding:8px 14px;border-radius:6px;display:none;max-width:70%;font:12px monospace;white-space:pre-wrap"></div>
<script>window.addEventListener('error',e=>{const d=document.getElementById('err');d.style.display='block';d.textContent='JS error: '+e.message+' ('+(e.filename||'').split('/').pop()+':'+e.lineno+')';});</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const LIVE = __LIVE__;            // true when served by tools/live_viewer.py (polls for new training events)
const RUNS = __RUNS__;
const F = {t:0,x:1,z:2,h:3,roll:4,pitch:5,hdg:6,kt:7,vs:8,thr:9,ele:10,ail:11,rud:12};
const R2D = Math.PI/180;

const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight); renderer.shadowMap.enabled = true; document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x9cc4e4); scene.fog = new THREE.Fog(0x9cc4e4, 600, 9000);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.5, 40000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x4a6b3a, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(-400, 700, 300); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, {left:-150,right:150,top:150,bottom:-150,near:10,far:2500}); scene.add(sun, sun.target);

// ---------- ground texture ----------
function grassTexture(){ const c=document.createElement('canvas'); c.width=c.height=128; const g=c.getContext('2d');
  g.fillStyle='#4f7a3a'; g.fillRect(0,0,128,128); g.fillStyle='#4a7336'; g.fillRect(0,0,64,64); g.fillRect(64,64,64,64);
  for(let i=0;i<500;i++){g.fillStyle=`rgba(${60+Math.random()*30},${100+Math.random()*40},50,.25)`;g.fillRect(Math.random()*128,Math.random()*128,2,2);}
  const t=new THREE.CanvasTexture(c); t.wrapS=t.wrapT=THREE.RepeatWrapping; return t; }
const gt = grassTexture(); gt.repeat.set(600, 60);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(60000, 6000), new THREE.MeshLambertMaterial({map:gt}));
ground.rotation.x = -Math.PI/2; ground.position.set(20000, -0.05, 0); ground.receiveShadow = true; scene.add(ground);

// ---------- runway (rebuilt per run) ----------
let runwayGroup = null;
function box(w,h,d,color,x,y,z){ const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d), new THREE.MeshLambertMaterial({color})); m.position.set(x,y,z); m.receiveShadow=true; return m; }
function buildRunway(L, W){
  if(runwayGroup) scene.remove(runwayGroup); runwayGroup = new THREE.Group();
  runwayGroup.add(box(L, 0.1, W, 0x3b3b3f, L/2, 0.0, 0));            // asphalt
  runwayGroup.add(box(L, 0.12, 1.0, 0xffffff, L/2, 0.03, -W/2+1.2), box(L, 0.12, 1.0, 0xffffff, L/2, 0.03, W/2-1.2));
  for(let x=60; x<L-80; x+=60) runwayGroup.add(box(30, 0.12, 0.9, 0xffffff, x, 0.04, 0));
  for(let i=-5;i<=5;i++){ if(i===0) continue; runwayGroup.add(box(30,0.12,1.6,0xffffff,25,0.04,i*5.0)); }       // threshold bars
  for(const d of [300,450,600]) for(const s of [-1,1]) runwayGroup.add(box(22,0.12,3.0,0xffffff,d,0.04,s*9));    // aiming/touchdown marks
  runwayGroup.add(box(1.5,0.14,W,0xdd2222,L,0.05,0));                                                           // end of runway
  const mk = new THREE.MeshLambertMaterial({color:0xffcc00});
  for(let x=500; x<=L; x+=500){ for(const s of [-1,1]){ const p=new THREE.Mesh(new THREE.BoxGeometry(1,5,6),mk); p.position.set(x,2.5,s*(W/2+14)); runwayGroup.add(p);} }
  // trees as speed cues (deterministic pseudo-random)
  const tg = new THREE.ConeGeometry(5, 18, 7), tm = new THREE.MeshLambertMaterial({color:0x2e5a2a});
  const trees = new THREE.InstancedMesh(tg, tm, 900); let s=12345; const rnd=()=> (s=(s*1664525+1013904223)%4294967296)/4294967296;
  const m4=new THREE.Matrix4();
  for(let i=0;i<900;i++){ const x=-300+rnd()*(L+4000), side=rnd()<.5?-1:1, z=side*(110+rnd()*700), sc=.6+rnd()*1.2;
    m4.compose(new THREE.Vector3(x,9*sc,z), new THREE.Quaternion(), new THREE.Vector3(sc,sc,sc)); trees.setMatrixAt(i,m4); }
  runwayGroup.add(trees);
  scene.add(runwayGroup);
}

// ---------- aircraft (visual stand-in; geometry positions from the FDM where available) ----------
const ac = new THREE.Group(); const acPivot = new THREE.Group(); acPivot.add(ac); scene.add(acPivot);
const white = new THREE.MeshPhongMaterial({color:0xf2f2f2, shininess:60}), grey = new THREE.MeshPhongMaterial({color:0xbfc6cc}),
      dark = new THREE.MeshPhongMaterial({color:0x2a2f35}), red = new THREE.MeshPhongMaterial({color:0xc02020});
(function build(){
  // fuselage: lathe profile (radius vs station), axis along +X, nose at +36 m, tail at -40 m
  const prof=[[0.0,36],[1.3,35],[2.4,32],[3.0,28],[3.3,22],[3.3,-12],[3.0,-22],[2.2,-31],[1.0,-38],[0.2,-40]].map(([r,x])=>new THREE.Vector2(r,x));
  const fus=new THREE.Mesh(new THREE.LatheGeometry(prof,28), white); fus.rotation.z=-Math.PI/2; fus.castShadow=true; ac.add(fus);
  const hump=new THREE.Mesh(new THREE.SphereGeometry(1,20,14), white); hump.scale.set(16,2.4,2.6); hump.position.set(22,3.1,0); hump.castShadow=true; ac.add(hump);
  const stripe=new THREE.Mesh(new THREE.CylinderGeometry(3.31,3.31,2.2,28,1,true), red); stripe.rotation.z=Math.PI/2; stripe.position.set(2,0,0); ac.add(stripe);
  const cock=new THREE.Mesh(new THREE.BoxGeometry(2.2,.7,3.3), dark); cock.position.set(30.5,2.4,0); ac.add(cock);
  const toMesh=(geo)=>{ geo.applyMatrix4(new THREE.Matrix4().set(0,1,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,1)); return geo; };
  const wingShape=new THREE.Shape([[-34.2,-19],[0,5],[34.2,-19],[34.2,-23],[0,-13],[-34.2,-23]].map(([s,t])=>new THREE.Vector2(s,t)));
  const wing=new THREE.Mesh(toMesh(new THREE.ExtrudeGeometry(wingShape,{depth:.6,bevelEnabled:false})), grey); wing.position.set(0,-1.6,0); wing.castShadow=true; ac.add(wing);
  const tailShape=new THREE.Shape([[-10,-32],[0,-28],[10,-32],[10,-35.5],[0,-33],[-10,-35.5]].map(([s,t])=>new THREE.Vector2(s,t)));
  const hs=new THREE.Mesh(toMesh(new THREE.ExtrudeGeometry(tailShape,{depth:.35,bevelEnabled:false})), grey); hs.position.set(0,1.0,0); ac.add(hs);
  const fin=new THREE.Shape([[-24,1.5],[-37,1.5],[-39,13.5],[-33.5,13.5]].map(([x,y])=>new THREE.Vector2(x,y)));
  const finM=new THREE.Mesh(new THREE.ExtrudeGeometry(fin,{depth:.6,bevelEnabled:false}), red); finM.position.z=-.3; finM.castShadow=true; ac.add(finM);
  // engines from FDM station data (x aft-positive in FDM, CG at 1327 in): inner +8.4 m fwd, y +-11.7 m ; outer -0.7 m, +-20.8 m
  for(const [x,z,y] of [[8.4,-11.7,-2.9],[8.4,11.7,-2.9],[-0.7,-20.8,-2.5],[-0.7,20.8,-2.5]]){
    const e=new THREE.Mesh(new THREE.CylinderGeometry(1.35,1.2,5.4,20), grey); e.rotation.z=Math.PI/2; e.position.set(x+1.5,y,z); e.castShadow=true; ac.add(e);
    const f=new THREE.Mesh(new THREE.CircleGeometry(1.2,20), dark); f.rotation.y=Math.PI/2; f.position.set(x+4.25,y,z); ac.add(f); }
  // landing gear (FDM stations: nose +23.6 m fwd; mains -5.8 m aft, +-5.5 m lateral) - wheel contact at -CG height
  for(const [x,z] of [[23.6,0],[-5.8,-5.5],[-5.8,5.5],[-7.6,-5.5],[-7.6,5.5]]){
    const w=new THREE.Mesh(new THREE.CylinderGeometry(.62,.62,.55,14), dark); w.rotation.x=Math.PI/2; w.position.set(x,-3.2,z); ac.add(w);
    const st=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,2.6), dark); st.position.set(x,-2,z); ac.add(st); }
})();

// ---------- optional detailed GLB model (embedded as base64) ----------
// Measured for 'Boeing747ERF' (manilov.ap, CC-BY-4.0): wings along Z, up = +Y, nose toward -X, origin not centred.
// So: yaw 180 deg, centre the span, put the CG GLB_CG_FROM_NOSE_M behind the nose, lowest point -> wheel contact,
// uniform scale so the span equals the FDM's 747-8 span (68.4 m).
const GLB_B64 = document.getElementById('glb').textContent.trim(); const GLB_CG_FROM_NOSE_M = __GLB_CG__; const FDM_SPAN_M = 68.4;
let glbRoot = null; const simpleParts = ac.children.slice();
function setModel(useGlb){ simpleParts.forEach(c=>c.visible=!useGlb); if(glbRoot) glbRoot.visible=useGlb; document.getElementById('mdl').value=useGlb?'glb':'simple'; }
function loadGlb(bin){
  document.getElementById('credit').style.display='block';
  import('three/addons/loaders/GLTFLoader.js').then(({GLTFLoader})=>{
    new GLTFLoader().parse(bin,'',(gltf)=>{
      const model=gltf.scene; model.updateMatrixWorld(true); const box=new THREE.Box3().setFromObject(model); const size=box.getSize(new THREE.Vector3());
      const s=FDM_SPAN_M/size.z; const wrap=new THREE.Group();
      model.position.set(-(box.min.x+GLB_CG_FROM_NOSE_M), -(box.min.y), -(box.min.z+box.max.z)/2);   // CG at origin (x), span centred (z), wheels at y=0
      wrap.add(model); wrap.scale.setScalar(s); wrap.rotation.y=Math.PI;
      // Coming from the FDM: at rest the body attitude is roughly -3 deg nose-down because the
      // nose gear springs compress.  Our GLB inherits that attitude in state; feeding in a small
      // positive x orientation makes the visible wheel bottom come off the ground at the same point
      // as in the real B747, instead of showing the front gear below orifice level.
      wrap.rotation.x = THREE.MathUtils.degToRad(3.2);   // nose W up a little, so the from-wheekcontact -> CG works in both rest and initial flight
      const holder=new THREE.Group(); holder.add(wrap); holder.position.y=0; glbRoot=holder; glbRoot.userData.s=s;
      model.traverse(o=>{ if(o.isMesh){ o.castShadow=true; o.receiveShadow=false; if(o.material){ o.material.side=THREE.FrontSide; } } });
      ac.add(glbRoot); glbHolderY(); setModel(true);
    },(e)=>{ const d=document.getElementById('err'); d.style.display='block'; d.textContent='GLB load failed: '+e; });
  });
}
if(GLB_B64){ loadGlb(Uint8Array.from(atob(GLB_B64),ch=>ch.charCodeAt(0)).buffer); }
else if(LIVE){ fetch('/model.glb').then(r=>r.ok?r.arrayBuffer():null).then(b=>{ if(b) loadGlb(b); }); }
function glbHolderY(){ if(glbRoot) glbRoot.position.y=-currentCgHeight; }   // wheel contact is cg_height below the CG
let currentCgHeight = 3.8;

// ---------- trail ----------
let trail = null;
function buildTrail(run){ if(trail) scene.remove(trail);
  const pts=run.frames.map(f=>new THREE.Vector3(f[F.x], f[F.h]+0.6, f[F.z]));
  trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:run.success?0x22dd66:0xff4444})); scene.add(trail); }

// ---------- playback ----------
let run = RUNS[0] || null, T = 0, playing = true, speed = 1, camMode = 0, last = performance.now();
const sel = document.getElementById('run');
function addOption(r,i){ const o=document.createElement('option'); o.value=i; o.textContent=`${i+1}. ${r.label} -> ${r.success?'SUCCESS':r.reason}`; sel.appendChild(o); }
RUNS.forEach(addOption);
const BADGE = {FIRST_SUCCESS:'\u2605', TRAINING_SUCCESS:'\u2605', EVAL_FIRST_SUCCESS:'\u2605', PAST_SUCCESS:'\u2606', STEP_UPDATE:'\u25C6', BIG_PROGRESS:'\u25B2', LEVEL_UP:'\u25B2', FIRST_LIFTOFF:'\u25B2', CHECKPOINT:'\u25C6'};
function renderEvents(){ if(!LIVE) return; const box=document.getElementById('events'); box.style.display='block'; box.innerHTML='';
  for(let i=RUNS.length-1;i>=Math.max(0,RUNS.length-60);i--){ const r=RUNS[i]; const d=document.createElement('div'); if(r===run) d.className='cur';
    d.style.color=r.success?'#8f8':(r.event_type==='BIG_PROGRESS'||r.event_type==='LEVEL_UP'||r.event_type==='FIRST_LIFTOFF'?'#fd8':'#fff');
    d.textContent=`${BADGE[r.event_type]||'\u2022'} ${r.run_name||''} ${((r.num_timesteps||0)/1000).toFixed(0)}k ${shortTitle(r)}`; d.title=r.headline||r.label; d.onclick=()=>{ pending=[]; selectRun(i); }; box.appendChild(d); } }
function shortTitle(r){ if(r.event_type==='PAST_SUCCESS') return `PAST success${r.reconstructed?' (reconstructed)':''}`;
  const h=r.headline||r.label||''; return h.length>58? h.slice(0,57)+'\u2026' : h; }
let toastTimer=null; function notify(r){ const t=document.getElementById('toast'); t.textContent=(BADGE[r.event_type]||'')+' '+(r.headline||r.label); t.style.display='block'; clearTimeout(toastTimer); toastTimer=setTimeout(()=>t.style.display='none',7000); }
let pending=[], endHold=0; const known=new Set(); let firstPoll=true;
async function poll(){ const st=document.getElementById('status');
  try{
    const list=await (await fetch('/api/events')).json(); const added=[];
    for(const e of list){ if(known.has(e.event_id)) continue; known.add(e.event_id);
      try{ const r=await (await fetch(e.url)).json(); const idx=RUNS.push(r)-1; addOption(r,idx); added.push(idx);} catch(err){ known.delete(e.event_id); } }
    if(added.length){
      if(firstPoll){ selectRun(added[added.length-1]); }
      else{ notify(RUNS[added[0]]); if(document.getElementById('follow').checked){ selectRun(added[0]); pending=added.slice(1); } }
      renderEvents(); }
    firstPoll=false;
    const s=await (await fetch('/api/status')).json(); st.style.display='block';
    st.innerHTML='<b>Live training status</b> (updates every few seconds)<br>'+(Object.keys(s).length?Object.entries(s).map(([name,v])=>{
      const roll=v.rolling_success_rate==null?'n/a':(100*v.rolling_success_rate).toFixed(0)+'%'; const ev=v.eval?(100*v.eval.success_rate).toFixed(0)+'% @'+(v.eval.num_timesteps/1000).toFixed(0)+'k':'n/a';
      const idle=(Date.now()/1000-(v.wall_time||0))>90;   // status file not refreshed for 90 s -> run is not training now
      return `<b>${name}</b>${idle?' <span style="color:#fa6">[idle/stopped]</span>':' <span style="color:#6f6">[training]</span>'}: ${(v.num_timesteps/1000).toFixed(0)}k transitions | level ${v.level} | training successes ${v.training_successes}/${v.episodes} episodes (last ${v.rolling_window}: ${roll}) | eval ${ev} | first success: ${v.steps_to_first_success==null?'NOT YET':(v.steps_to_first_success/1000).toFixed(0)+'k'}`; }).join('<br>'):'no runs with showcase data yet');
  }catch(err){ st.style.display='block'; st.textContent='Live viewer server not reachable ('+err+')'; }
}
if(LIVE){ document.getElementById('followlbl').style.display='inline'; poll(); setInterval(poll,3000); }
const orbit = new OrbitControls(camera, renderer.domElement); orbit.enableDamping = true;
function pose(t){ const fr=run.frames, dt=fr[1][F.t]-fr[0][F.t]; t=Math.max(0,t); let i=Math.max(0,Math.min(Math.floor(t/dt), fr.length-2)), a=fr[i], b=fr[i+1], u=Math.max(0,Math.min(1,(t-a[F.t])/dt));
  const o={}; for(const k in F) o[k]=a[F[k]]+(b[F[k]]-a[F[k]])*u; return o; }
function duration(){ return run.frames[run.frames.length-1][F.t]; }
function selectRun(i){ run=RUNS[i]; sel.value=i; currentCgHeight=run.cg_height_m; glbHolderY(); T=0; playing=true; endHold=0; document.getElementById('play').textContent='Pause'; buildRunway(run.runway_length_m, run.runway_width_m); buildTrail(run);
  const ttl=document.getElementById('title'); ttl.textContent=(BADGE[run.event_type]||'')+' '+(run.run_name?run.run_name+' @ '+((run.num_timesteps||0)/1000).toFixed(0)+'k: ':'')+shortTitle(run); ttl.title=run.headline||run.label;
  document.getElementById('cond').textContent=`${run.info} | runway ${run.runway_length_m.toFixed(0)} m`+(run.reconstructed?' | re-flown under the original conditions (original trajectory not saved)':'');
  document.getElementById('banner').style.display='none'; camPos.set(-120,25,0); camSet=false; renderEvents(); }
sel.onchange=()=>{ pending=[]; selectRun(+sel.value); };
document.getElementById('play').onclick=()=>{playing=!playing; document.getElementById('play').textContent=playing?'Pause':'Play';};
document.getElementById('restart').onclick=()=>{T=0; playing=true; document.getElementById('banner').style.display='none';};
document.getElementById('mdl').onchange=e=>setModel(e.target.value==='glb');
document.getElementById('spd').onchange=e=>speed=+e.target.value; document.getElementById('cam').onchange=e=>camMode=+e.target.value;
const seek=document.getElementById('seek'); seek.oninput=()=>{T=seek.value/1000*duration();};
addEventListener('keydown',e=>{ if(e.key==='h'||e.key==='H') document.body.classList.toggle('hide-ui');   // H: hide/show all overlays
  if(e.code==='Space'){e.preventDefault();document.getElementById('play').click();} if('12345'.includes(e.key)){camMode=+e.key-1;document.getElementById('cam').value=camMode;}
  if(e.key==='r'||e.key==='R')document.getElementById('restart').click(); const s=document.getElementById('spd'); if(e.key==='+'&&s.selectedIndex<4)s.selectedIndex++; if(e.key==='-'&&s.selectedIndex>0)s.selectedIndex--; speed=+s.value; });
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});

const camPos=new THREE.Vector3(-120,25,0); let camSet=false; const camTgt=new THREE.Vector3();
const bar=(id,v,lo,hi)=>{const el=document.getElementById(id); const a=(Math.min(hi,Math.max(lo,v))-lo)/(hi-lo); if(lo<0){const m=.5; el.style.left=(Math.min(a,m)*100)+'%'; el.style.width=(Math.abs(a-m)*100)+'%';} else {el.style.left='0'; el.style.width=(a*100)+'%';}};
function frame(){ const now=performance.now(); const dtS=Math.max(0,Math.min((now-last)/1000,0.1)); last=now;
  if(!run){ acPivot.visible=false; camera.position.set(-150,40,120); camera.lookAt(250,0,0);
    document.getElementById('tab').innerHTML='<tr><td colspan="2">waiting for the first training event...</td></tr>'; renderer.render(scene,camera); requestAnimationFrame(frame); return; }
  acPivot.visible=true;
  if(playing){ T+=dtS*speed; if(T>=duration()){T=duration(); playing=false; document.getElementById('play').textContent='Play';} }
  if(LIVE && !playing && T>=duration()-1e-6 && document.getElementById('follow').checked){   // playlist: next queued event, else replay the current one
    if(!endHold) endHold=now; if(now-endHold>4500){ endHold=0; if(pending.length){ selectRun(pending.shift()); } else { T=0; playing=true; document.getElementById('play').textContent='Pause'; } } }
  const p=pose(T), yaw=p.hdg*R2D, cgy=p.h+run.cg_height_m;
  acPivot.position.set(p.x, cgy, p.z); ac.rotation.set(p.roll*R2D, -yaw, p.pitch*R2D, 'YZX');
  sun.position.set(p.x-400, 700, p.z+300); sun.target.position.set(p.x,0,p.z);
  const f=new THREE.Vector3(Math.cos(yaw),0,Math.sin(yaw)), pos=acPivot.position;
  if(camMode===3){ if(!camSet){camera.position.set(p.x-110,40,90); orbit.target.copy(pos); camSet=true;} const d=pos.clone().sub(orbit.target); orbit.target.add(d); camera.position.add(d); orbit.update(); }
  else{ let want, look=pos.clone(); camSet=false;
    if(camMode===0){ want=pos.clone().addScaledVector(f,-110).add(new THREE.Vector3(0,26,0)); look.addScaledVector(f,25); }
    else if(camMode===1){ want=pos.clone().add(new THREE.Vector3(-10,12,150)); }
    else if(camMode===2){ want=new THREE.Vector3(Math.min(Math.max(p.x,250),run.runway_length_m*0.55),22,95); }
    else { want=pos.clone().add(new THREE.Vector3(0,400,0.01)); }
    camPos.lerp(want,1-Math.exp(-dtS*(camMode===0?5:camMode===4?8:3))); camera.position.copy(camPos); camTgt.lerp(look,1-Math.exp(-dtS*8)); camera.lookAt(camTgt); }
  document.getElementById('tab').innerHTML=
   `<tr><td>time</td><td>${p.t.toFixed(1)} s</td></tr><tr><td>airspeed</td><td>${p.kt.toFixed(0)} kt (VR est ${run.vr_kt.toFixed(0)})</td></tr>
    <tr><td>height AGL</td><td>${(p.h*3.2808).toFixed(0)} ft</td></tr><tr><td>vertical speed</td><td>${p.vs.toFixed(0)} fpm</td></tr>
    <tr><td>pitch / roll</td><td>${p.pitch.toFixed(1)}&deg; / ${p.roll.toFixed(1)}&deg;</td></tr>
    <tr><td>along runway</td><td>${p.x.toFixed(0)} m</td></tr><tr><td>centerline offset</td><td style="color:${Math.abs(p.z)>25?'#f66':'#fff'}">${p.z.toFixed(1)} m</td></tr>
    <tr><td>heading error</td><td>${p.hdg.toFixed(1)}&deg;</td></tr>`;
  bar('b_thr',p.thr,0,1); bar('b_ele',p.ele,-1,1); bar('b_ail',p.ail,-1,1); bar('b_rud',p.rud,-1,1);
  document.getElementById('clock').textContent=`${T.toFixed(1)} / ${duration().toFixed(1)} s`; seek.value=T/duration()*1000;
  const bn=document.getElementById('banner'); if(T>=duration()-1e-6){ bn.style.display='block'; bn.textContent=run.success?'SUCCESS: stable, clean takeoff above 1,000 ft AGL for 5 s':'FAILED: '+run.reason; bn.style.background=run.success?'rgba(20,140,60,.9)':'rgba(180,30,30,.9)'; }
  else bn.style.display='none';
  renderer.render(scene,camera); requestAnimationFrame(frame); }
if(RUNS.length) selectRun(0);
if(run){ // optional URL hash, e.g. #run=3&t=38&cam=1&pause=1  (jump to a moment; useful for screenshots)
  const q=new URLSearchParams(location.hash.slice(1));
  if(q.has('run')){ sel.value=q.get('run'); selectRun(+q.get('run')); }
  if(q.has('cam')){ camMode=+q.get('cam'); document.getElementById('cam').value=camMode; }
  if(q.has('t')) T=+q.get('t');
  if(q.get('pause')==='1'){ playing=false; document.getElementById('play').textContent='Play'; }
  camTgt.copy(pose(T) ? new THREE.Vector3(pose(T).x, pose(T).h, pose(T).z) : camTgt); camPos.set(pose(T).x-110, 30, pose(T).z);
}
frame();   // render the first frame immediately (also reveals any runtime error)
window.__ready = true;
</script></body></html>
"""

if __name__ == "__main__":
    main()
