"""Generate a self-contained interactive HTML report for a cluster run.

Used by the standalone CLI (`--viz out.html`) so a result can be explored in a
browser with no server. The page inlines the result JSON and renders charts via
plotly.js from a CDN.
"""
# ruff: noqa: E501 - this module embeds a JS template with long lines

from __future__ import annotations

import json

from embedrag.cluster.types import ClusterResult

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cluster Report - {run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f1117; color:#e6e6e6; }}
  header {{ padding: 16px 24px; background:#161922; border-bottom:1px solid #2a2f3a; }}
  h1 {{ font-size: 18px; margin: 0; }}
  .meta {{ color:#9aa4b2; font-size: 13px; margin-top: 4px; }}
  main {{ padding: 16px 24px; display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}
  .panel {{ background:#161922; border:1px solid #2a2f3a; border-radius:8px; padding:12px; }}
  .panel.full {{ grid-column: 1 / -1; }}
  .panel h2 {{ font-size: 14px; margin:0 0 8px; color:#c9d1d9; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #2a2f3a; vertical-align: top; }}
  .kw {{ color:#7ee787; }}
  .chip {{ display:inline-block; background:#21262d; border-radius:10px; padding:1px 8px; margin:2px; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>Cluster Report</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="panel full"><h2>2D projection</h2><div id="scatter" style="height:480px;"></div></div>
  <div class="panel"><h2>Cluster sizes</h2><div id="sizes" style="height:320px;"></div></div>
  <div class="panel" id="auxPanel"><h2 id="auxTitle">Detail</h2><div id="aux" style="height:320px;"></div></div>
  <div class="panel full"><h2>Clusters</h2><div id="table"></div></div>
</main>
<script>
const RESULT = {result_json};
</script>
<script>{render_js}</script>
</body>
</html>
"""

# Shared renderer (also referenced conceptually by the served page).
_RENDER_JS = r"""
function palette(i){const c=['#58a6ff','#7ee787','#ff7b72','#d2a8ff','#ffa657','#79c0ff','#f778ba','#56d364','#e3b341','#a5d6ff'];return i<0?'#6e7681':c[i%c.length];}
function renderReport(R){
  document.getElementById('meta').textContent =
    `${R.algorithm} | ${R.n_clusters} clusters | ${R.noise_count} noise | ${R.n_items} items | silhouette ${R.metrics.silhouette}`;
  const P=R.projection||{};
  if(P.x){
    const byC={};
    for(let i=0;i<P.x.length;i++){const c=P.cluster[i];(byC[c]=byC[c]||{x:[],y:[],t:[]});byC[c].x.push(P.x[i]);byC[c].y.push(P.y[i]);byC[c].t.push((P.id?P.id[i]:'')+': '+(P.text?P.text[i]:''));}
    const traces=Object.keys(byC).map(c=>({x:byC[c].x,y:byC[c].y,text:byC[c].t,mode:'markers',type:'scattergl',name:c==-1?'noise':('cluster '+c),marker:{size:6,color:palette(parseInt(c))}}));
    Plotly.newPlot('scatter',traces,{paper_bgcolor:'#161922',plot_bgcolor:'#161922',font:{color:'#e6e6e6'},margin:{t:10},legend:{orientation:'h'}},{responsive:true});
  }
  const sizes=R.clusters.map(c=>c.size), names=R.clusters.map(c=>c.label||('cluster '+c.cluster_id));
  Plotly.newPlot('sizes',[{x:names,y:sizes,type:'bar',marker:{color:R.clusters.map(c=>palette(c.cluster_id))}}],{paper_bgcolor:'#161922',plot_bgcolor:'#161922',font:{color:'#e6e6e6'},margin:{t:10,b:120}},{responsive:true});
  renderAux(R);
  renderTable(R);
}
function renderAux(R){
  const kcurve=(R.viz||[]).find(v=>v.type==='k_curve');
  const sil=(R.viz||[]).find(v=>v.type==='silhouette');
  const sim=(R.viz||[]).find(v=>v.type==='similarity_matrix');
  if(kcurve){document.getElementById('auxTitle').textContent='Score vs K';Plotly.newPlot('aux',[{x:kcurve.data.k,y:kcurve.data.silhouette,mode:'lines+markers',name:'silhouette'}],{paper_bgcolor:'#161922',plot_bgcolor:'#161922',font:{color:'#e6e6e6'},margin:{t:10}},{responsive:true});}
  else if(sim&&sim.data.matrix.length){document.getElementById('auxTitle').textContent='Inter-cluster similarity';Plotly.newPlot('aux',[{z:sim.data.matrix,x:sim.data.cluster_ids,y:sim.data.cluster_ids,type:'heatmap',colorscale:'Blues'}],{paper_bgcolor:'#161922',plot_bgcolor:'#161922',font:{color:'#e6e6e6'},margin:{t:10}},{responsive:true});}
  else{document.getElementById('auxPanel').style.display='none';}
}
function renderTable(R){
  let h='<table><tr><th>ID</th><th>Label</th><th>Size</th><th>Cohesion</th><th>Keywords</th><th>Example</th></tr>';
  for(const c of R.clusters){
    const kw=(c.keywords||[]).slice(0,8).map(k=>`<span class="chip kw">${esc(k)}</span>`).join(' ');
    const ex=(c.representative_texts&&c.representative_texts[0])?esc(c.representative_texts[0].slice(0,160)):'';
    h+=`<tr><td>${c.cluster_id}</td><td>${esc(c.label||'')}<div style="color:#9aa4b2">${esc(c.summary||'')}</div></td><td>${c.size}</td><td>${c.cohesion}</td><td>${kw}</td><td>${ex}</td></tr>`;
  }
  h+='</table>';
  document.getElementById('table').innerHTML=h;
}
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
if(typeof RESULT!=='undefined'){renderReport(RESULT);}
"""


def write_html(result: ClusterResult, path: str) -> None:
    """Write a self-contained interactive HTML report."""
    html = _TEMPLATE.format(
        run_id=result.run_id,
        result_json=json.dumps(result.to_dict(), ensure_ascii=False),
        render_js=_RENDER_JS,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def render_js() -> str:
    """Return the shared render JS (reused by the served /cluster page)."""
    return _RENDER_JS
