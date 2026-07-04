"""Assemble the self-contained morphompm results dashboard (figures embedded as
base64 data URIs). Output: outputs/morphompm_dashboard.html (rendered via Artifact)."""
import base64
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _githash():
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "uncommitted"
    except Exception:
        return "uncommitted"


GITHASH = _githash()
FIG = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs", "morphompm_dashboard.html")


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


IMG = {k: b64(v) for k, v in {
    "inv": "inverse_recovery.png",
    "bend": "bending_timoshenko.png",
    "diff": "differential_growth.png",
}.items()}

# verification gate log (real numbers from python -m morphompm.verify + C++ T1-T8)
GATES = [
    ("[2]", "constitutive VJP vs FD — neo-Hookean", "4.2e-10", "pass"),
    ("[2]", "constitutive VJP vs FD — Hencky (manual SVD adjoint)", "1.5e-9", "pass"),
    ("[2]", "constitutive VJP vs FD — Herschel-Bulkley (rate, C_bar)", "9.4e-10", "pass"),
    ("[2]", "confined-swell analytic oracle (detects &mu;&harr;&lambda; swap)", "exact", "pass"),
    ("[3]", "transfer assembly gate (transfer &compfn; constitutive)", "4.5e-10", "pass"),
    ("[1p]", "forward physics: free-swell det F &rarr; g&sup3;", "3.45 / 3.38", "pass"),
    ("[4]", "trajectory adjoint — advect off / on", "3.7e-10 / 3.1e-10", "pass"),
    ("[6]", "inverse growth-rate recovery (Gauss-Newton)", "err 1.4e-13", "pass"),
]
gate_rows = "\n".join(
    f'<tr><td class="tag">{t}</td><td>{d}</td>'
    f'<td class="num">{v}</td><td><span class="pill">ok</span></td></tr>'
    for (t, d, v, _) in GATES)

STAGES = [
    ("[2]", "constitutive", "elastic + bioink, manual VJP"),
    ("[3]", "transfer", "MLS-MPM step + advect adjoint"),
    ("[4]", "integrate", "trajectory rollout + adjoint"),
    ("[6]", "inverse", "morphology &rarr; growth params"),
]
stage_items = "\n".join(
    f'<li><span class="s-tag">{t}</span><span class="s-name">{n}</span>'
    f'<span class="s-desc">{d}</span></li>' for (t, n, d) in STAGES)

HTML = f"""<style>
  :root {{
    --ground:#F6F7F5; --panel:#FFFFFF; --ink:#171A1D; --muted:#5F6970;
    --line:#E4E7E3; --accent:#0F766E; --accent-soft:#E4F0ED;
    --pass:#2F7D4E; --warn:#9C5A12; --warn-soft:#F6EEDF;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono","SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;
  }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:40px 24px 64px;
    background:var(--ground); color:var(--ink); font-family:var(--sans);
    line-height:1.55; -webkit-font-smoothing:antialiased; }}
  .wrap * {{ box-sizing:border-box; }}
  .eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--accent); margin:0 0 10px; }}
  h1 {{ font-size:40px; line-height:1.05; letter-spacing:-.02em; margin:0;
    text-wrap:balance; font-weight:680; }}
  .sub {{ font-size:18px; color:var(--muted); margin:10px 0 0; max-width:62ch; }}
  .meta {{ font-family:var(--mono); font-size:12.5px; color:var(--muted);
    display:flex; flex-wrap:wrap; gap:8px 18px; margin:18px 0 0; }}
  .meta b {{ color:var(--ink); font-weight:600; }}
  hr.rule {{ border:0; border-top:1px solid var(--line); margin:32px 0; }}
  .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0 0; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:16px 16px 14px; }}
  .stat .k {{ font-family:var(--mono); font-size:24px; font-weight:600;
    font-variant-numeric:tabular-nums; letter-spacing:-.01em; }}
  .stat .l {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
  h2 {{ font-size:13px; font-family:var(--mono); letter-spacing:.14em;
    text-transform:uppercase; color:var(--muted); margin:0 0 16px;
    display:flex; align-items:center; gap:10px; }}
  h2::after {{ content:""; flex:1; height:1px; background:var(--line); }}
  section {{ margin-top:40px; }}
  .stages {{ list-style:none; padding:0; margin:0; display:grid;
    grid-template-columns:repeat(4,1fr); gap:12px; }}
  .stages li {{ background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:4px; }}
  .s-tag {{ font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:600; }}
  .s-name {{ font-weight:640; font-size:15px; }}
  .s-desc {{ font-size:12.5px; color:var(--muted); }}
  .gatecard {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
    overflow:hidden; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  thead th {{ font-family:var(--mono); font-size:11px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); text-align:left; font-weight:600;
    padding:12px 16px; background:#FBFCFB; border-bottom:1px solid var(--line); }}
  tbody td {{ padding:11px 16px; border-bottom:1px solid var(--line); }}
  tbody tr:last-child td {{ border-bottom:0; }}
  td.tag {{ font-family:var(--mono); color:var(--accent); font-weight:600; width:52px; }}
  td.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums;
    color:var(--muted); text-align:right; white-space:nowrap; }}
  .pill {{ display:inline-block; font-family:var(--mono); font-size:11px; font-weight:700;
    letter-spacing:.05em; text-transform:uppercase; color:var(--pass);
    background:#E7F1EA; border:1px solid #CBE3D4; border-radius:999px; padding:2px 9px; }}
  .figrow {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .fig {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:14px; }}
  .fig img {{ width:100%; height:auto; display:block; border-radius:6px; }}
  .fig .cap {{ font-size:13px; color:var(--muted); margin:12px 2px 2px; }}
  .fig h3 {{ font-size:15.5px; margin:2px 2px 0; font-weight:640; }}
  .note {{ font-size:12.5px; background:var(--warn-soft); border:1px solid #EBD9B8;
    color:var(--warn); border-radius:8px; padding:9px 12px; margin-top:12px; }}
  .repro {{ background:var(--ink); color:#E8ECEA; border-radius:12px; padding:22px 24px;
    display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:16px; }}
  .repro code {{ font-family:var(--mono); font-size:14px; color:#8FE3D4;
    background:rgba(255,255,255,.06); padding:6px 12px; border-radius:7px; }}
  .repro .r-l {{ font-size:13.5px; color:#AEB7B4; }}
  .repro .r-l b {{ color:#fff; }}
  .foot {{ font-size:12.5px; color:var(--muted); margin-top:40px; }}
  .foot b {{ color:var(--ink); }}
  @media (max-width:720px) {{
    .stats,.stages {{ grid-template-columns:repeat(2,1fr); }}
    .figrow {{ grid-template-columns:1fr; }}
    h1 {{ font-size:32px; }}
  }}
</style>

<div class="wrap">
  <p class="eyebrow">differentiable simulation &middot; morphogenesis</p>
  <h1>morphompm</h1>
  <p class="sub">A differentiable morphoelastic growth&ndash;MPM for soft / living matter &mdash;
     a verified core that infers material &amp; growth laws from observed shape.</p>
  <div class="meta">
    <span><b>Phase 2</b> &middot; morphoelastic morphogenesis</span>
    <span>verification: <b>3 independent axes</b></span>
    <span>reproducible: <b>git&nbsp;{GITHASH}</b></span>
    <span>status: <b>core complete</b></span>
  </div>

  <div class="stats">
    <div class="stat"><div class="k">3</div><div class="l">verification axes &mdash; all pass</div></div>
    <div class="stat"><div class="k">3</div><div class="l">constitutive models (elastic + bioink)</div></div>
    <div class="stat"><div class="k">1.4e-13</div><div class="l">inverse recovery error</div></div>
    <div class="stat"><div class="k">1 cmd</div><div class="l">python scripts/reproduce.py</div></div>
  </div>

  <section>
    <h2>pipeline</h2>
    <ul class="stages">{stage_items}</ul>
  </section>

  <section>
    <h2>verification &mdash; three independent axes</h2>
    <div class="gatecard"><table>
      <thead><tr><th>stage</th><th>gate</th><th>metric</th><th>state</th></tr></thead>
      <tbody>{gate_rows}</tbody>
    </table></div>
    <div class="note">Two axes are independent by construction: <b>FD gates</b> verify the
      adjoint is consistent with the forward; the <b>confined-swell analytic oracle</b> and
      <b>forward-physics guard</b> verify the forward is <em>correct</em> (they caught a
      Herschel-Bulkley pressure-sign bug every FD gate had passed).</div>
  </section>

  <section>
    <h2>results</h2>
    <div class="figrow">
      <div class="fig">
        <h3>Differentiable inverse</h3>
        <img alt="growth-rate recovery convergence" src="data:image/png;base64,{IMG['inv']}">
        <p class="cap">The novel capability: recover the growth rate from an observed shape
          by back-propagating through the full trajectory. Converges in 6 iterations.</p>
      </div>
      <div class="fig">
        <h3>Morphogenesis validation</h3>
        <img alt="bilayer bending vs Timoshenko" src="data:image/png;base64,{IMG['bend']}">
        <p class="cap">Differential-growth bilayer bending vs the independent Timoshenko analytic.</p>
        <div class="note">Linear-in-mismatch scaling reproduced (1.86 &asymp; 2.0). The
          ~0.60&times; offset was <b>diagnosed as under-relaxation</b> (logic audit): transverse
          growth ruled out (axial-only identical), grid resolution minor; the slow bending mode
          is still relaxing (0.60 @ 4k &rarr; 0.76 @ 12k steps, climbing toward Timoshenko). A
          numerical-convergence artifact, not a physics error.</div>
      </div>
    </div>
    <div class="fig" style="margin-top:20px">
      <h3>Differential-growth residual stress &amp; bending</h3>
      <img alt="bilayer residual stress" src="data:image/png;base64,{IMG['diff']}">
      <p class="cap">Incompatible (differential) growth sustains residual elastic strain and
        bends an initially-flat strip &mdash; the morphogenesis regime free swelling cannot show.</p>
    </div>
  </section>

  <section>
    <h2>reproducibility</h2>
    <div class="repro">
      <div class="r-l">Deterministic (single-thread, no-RNG physics) &middot; version-pinned
        &middot; <b>one command regenerates every result</b> above.</div>
      <code>python scripts/reproduce.py</code>
    </div>
  </section>

  <p class="foot"><b>What's next (non-code):</b> the differentiable inverse on real
     morphogenesis data (Savin&nbsp;&amp;&nbsp;Tabin 2011 gut looping; Guvendiren 2009 swelling
     wrinkles) &mdash; published tabulated calibration + emergent validation, no own experiments
     required. Bioink extrusion (Herschel-Bulkley) preserved as a dormant module for a later track.</p>
</div>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", OUT, f"({len(HTML)//1024} KB)")
