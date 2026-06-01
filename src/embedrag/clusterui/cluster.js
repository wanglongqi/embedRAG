"use strict";

const PALETTE = ['#58a6ff','#7ee787','#ff7b72','#d2a8ff','#ffa657','#79c0ff','#f778ba','#56d364','#e3b341','#a5d6ff'];
function color(i){ return i < 0 ? '#6e7681' : PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]; }
function esc(s){ return (s == null ? '' : String(s)).replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }
function $(id){ return document.getElementById(id); }

let CURRENT = null;       // full result for the active run
let CURRENT_RUN_ID = null;

const PLOT_LAYOUT = { paper_bgcolor:'#161922', plot_bgcolor:'#161922', font:{color:'#e6e6e6'}, margin:{t:10} };

async function api(path, opts){
  const resp = await fetch(path, opts);
  if(!resp.ok){ const t = await resp.text(); throw new Error(`${resp.status}: ${t.slice(0,200)}`); }
  return resp.json();
}

async function loadVersion(){
  try {
    const r = await api('/readiness');
    $('versionBadge').textContent = r.active_version || '--';
    if (r.llm_url) {
      $('llmUrl').value = r.llm_url;
    }
    if (r.llm_model) {
      $('llmModel').value = r.llm_model;
    }
  }
  catch { $('versionBadge').textContent = 'offline'; }
}

async function refreshRuns(){
  try {
    const r = await api('/api/clusters');
    const pick = $('runPicker');
    pick.innerHTML = '<option value="">Saved runs…</option>';
    for(const run of r.runs){
      const o = document.createElement('option');
      o.value = run.run_id;
      o.textContent = `${run.run_id} · ${run.algorithm} · ${run.n_clusters} clusters`;
      pick.appendChild(o);
    }
  } catch(e){ /* ignore */ }
}

function buildRequest(){
  const params = {};
  const mcs = $('minClusterSize').value, k = $('kParam').value;
  if(mcs) params.min_cluster_size = parseInt(mcs);
  if(k){ params.k = parseInt(k); params.n_clusters = parseInt(k); }
  const eps = $('epsParam').value, ms = $('minSamples').value;
  if(eps) params.eps = parseFloat(eps);
  if(ms) params.min_samples = parseInt(ms);
  const sel = $('selectionMethod').value;
  if(sel) params.cluster_selection_method = sel;
  const link = $('linkage').value;
  if(link) params.linkage = link;
  const dt = $('distanceThreshold').value;
  if(dt) params.distance_threshold = parseFloat(dt);
  const res = $('leidenResolution').value, lknn = $('leidenKnn').value;
  if(res) params.resolution = parseFloat(res);
  if(lknn) params.knn = parseInt(lknn);

  const filters = {};
  if($('docType').value) filters.doc_type = $('docType').value.trim();
  if($('docId').value) filters.doc_id = $('docId').value.trim();
  return {
    algorithm: $('algorithm').value,
    reduce: $('reduce').value,
    auto: $('autoParam').checked,
    space: $('space').value || 'text',
    filters: Object.keys(filters).length ? filters : null,
    max_items: parseInt($('maxItems').value) || 20000,
    params,
    label_with_llm: $('useLlm').checked,
    llm_url: $('llmUrl').value.trim(),
    llm_model: $('llmModel').value.trim(),
    llm_language: $('llmLang').value.trim() || 'auto',
    persist: true,
  };
}

async function runClustering(){
  $('status').textContent = 'Clustering… (this can take a while)';
  $('runBtn').disabled = true;
  try {
    const body = buildRequest();
    const res = await api('/api/cluster', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body),
    });
    CURRENT = res; CURRENT_RUN_ID = res.run_id;
    $('status').textContent = `Done in ${Math.round(res.elapsed_ms)} ms`;
    render(res);
    refreshRuns();
  } catch(e){
    $('status').textContent = 'Error: ' + e.message;
  } finally {
    $('runBtn').disabled = false;
  }
}

async function loadRun(){
  const id = $('runPicker').value;
  if(!id) return;
  $('status').textContent = 'Loading run…';
  try {
    const res = await api('/api/clusters/' + encodeURIComponent(id));
    CURRENT = res; CURRENT_RUN_ID = res.run_id;
    $('status').textContent = 'Loaded ' + id;
    render(res);
  } catch(e){ $('status').textContent = 'Error: ' + e.message; }
}

function render(R){
  $('summary').style.display = 'flex';
  $('charts').style.display = 'grid';
  $('tablewrap').style.display = 'block';
  renderSummary(R);
  renderScatter(R);
  renderSizes(R);
  renderAux(R);
  renderTable(R);
}

function metricCard(k, v){ return `<div class="metric-card"><div class="v">${v == null ? '–' : v}</div><div class="k">${k}</div></div>`; }
function renderSummary(R){
  const m = R.metrics || {};
  $('summary').innerHTML = [
    metricCard('algorithm', esc(R.algorithm)),
    metricCard('clusters', R.n_clusters),
    metricCard('noise', R.noise_count),
    metricCard('items', R.n_items),
    metricCard('silhouette', m.silhouette),
    metricCard('davies-bouldin', m.davies_bouldin),
    metricCard('noise ratio', m.noise_ratio),
    R.metrics.external ? metricCard('ARI', R.metrics.external.ari) : '',
  ].join('');
}

function renderScatter(R){
  const P = R.projection || {};
  $('scatterHint').textContent = P.subsampled ? `(showing ${P.x.length} of ${P.total}, ${P.method})` : `(${P.method||''})`;
  if(!P.x || !P.x.length){ Plotly.purge('scatter'); return; }
  const byC = {};
  for(let i=0;i<P.x.length;i++){
    const c = P.cluster[i];
    (byC[c] = byC[c] || {x:[],y:[],t:[]});
    byC[c].x.push(P.x[i]); byC[c].y.push(P.y[i]);
    byC[c].t.push(`${esc(P.id ? P.id[i] : '')}: ${esc(P.text ? P.text[i] : '')}`);
  }
  const traces = Object.keys(byC).map(c => ({
    x: byC[c].x, y: byC[c].y, text: byC[c].t, hoverinfo:'text',
    mode:'markers', type: P.x.length > 2000 ? 'scattergl' : 'scatter',
    name: c == -1 ? 'noise' : ('cluster ' + c),
    marker:{ size:6, color: color(parseInt(c)), opacity:0.85 },
  }));
  const scatterDiv = $('scatter');
  Plotly.newPlot(scatterDiv, traces, Object.assign({}, PLOT_LAYOUT, {legend:{orientation:'h'}}), {responsive:true});

  // Cross-highlighting link
  scatterDiv.removeAllListeners && scatterDiv.removeAllListeners('plotly_click');
  scatterDiv.on('plotly_click', function(data){
    if(!data || !data.points || !data.points.length) return;
    const pt = data.points[0];
    const trace = scatterDiv.data[pt.curveNumber];
    if(!trace || !trace.name) return;
    if(trace.name === 'noise') return;
    const m = trace.name.match(/cluster (\d+)/);
    if(m) {
      highlightCluster(parseInt(m[1]));
    }
  });
}

function renderSizes(R){
  const names = R.clusters.map(c => c.label || ('cluster ' + c.cluster_id));
  const sizesDiv = $('sizes');
  Plotly.newPlot(sizesDiv, [{
    x: names, y: R.clusters.map(c => c.size), type:'bar',
    marker:{ color: R.clusters.map(c => color(c.cluster_id)) },
  }], Object.assign({}, PLOT_LAYOUT, {margin:{t:10,b:120}}), {responsive:true});

  // Cross-highlighting link
  sizesDiv.removeAllListeners && sizesDiv.removeAllListeners('plotly_click');
  sizesDiv.on('plotly_click', function(data){
    if(!data || !data.points || !data.points.length) return;
    const pt = data.points[0];
    if(CURRENT && CURRENT.clusters[pt.pointNumber]) {
      const cid = CURRENT.clusters[pt.pointNumber].cluster_id;
      highlightCluster(cid);
    }
  });
}

function renderAux(R){
  const viz = R.viz || [];
  const kcurve = viz.find(v => v.type === 'k_curve');
  const sil = viz.find(v => v.type === 'silhouette');
  const sim = viz.find(v => v.type === 'similarity_matrix');
  const dendro = viz.find(v => v.type === 'dendrogram');
  if(kcurve){
    $('auxTitle').textContent = 'Score vs K';
    Plotly.newPlot('aux', [
      { x:kcurve.data.k, y:kcurve.data.silhouette, mode:'lines+markers', name:'silhouette' },
    ], PLOT_LAYOUT, {responsive:true});
  } else if(sil){
    $('auxTitle').textContent = 'Silhouette by cluster';
    const traces = Object.keys(sil.data.per_cluster).map(c => ({
      y: sil.data.per_cluster[c], name:'c'+c, type:'box', marker:{color:color(parseInt(c))},
    }));
    Plotly.newPlot('aux', traces, PLOT_LAYOUT, {responsive:true});
  } else if(sim && sim.data.matrix && sim.data.matrix.length){
    $('auxTitle').textContent = 'Inter-cluster similarity';
    Plotly.newPlot('aux', [{
      z: sim.data.matrix, x: sim.data.cluster_ids, y: sim.data.cluster_ids,
      type:'heatmap', colorscale:'Blues', zmin:0, zmax:1,
    }], PLOT_LAYOUT, {responsive:true});
  } else if(dendro){
    $('auxTitle').textContent = 'Dendrogram (linkage available)';
    $('aux').innerHTML = '<p style="color:#9aa4b2;font-size:13px">Linkage matrix exported in the run JSON.</p>';
  } else {
    $('auxTitle').textContent = 'Detail';
    $('aux').innerHTML = '';
  }
}

function highlightCluster(cid){
  const tr = document.querySelector(`#clusterTable tr[data-cid="${cid}"]`);
  if(tr) {
    document.querySelectorAll('#clusterTable tr').forEach(r => r.classList.remove('highlighted-row'));
    tr.classList.add('highlighted-row');
    tr.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  openMembers(cid);
}

function renderTable(R){
  let h = '<table><thead><tr><th>ID</th><th>Label</th><th>Size</th><th>Cohesion</th><th>Sep.</th><th>Keywords</th></tr></thead><tbody>';
  for(const c of R.clusters){
    const kw = (c.keywords||[]).slice(0,8).map(k => `<span class="chip kw">${esc(k)}</span>`).join(' ');
    h += `<tr data-cid="${c.cluster_id}"><td>${c.cluster_id}</td>`
      + `<td>${esc(c.label||'')}<div class="hint">${esc(c.summary||'')}</div></td>`
      + `<td>${c.size}</td><td>${c.cohesion}</td><td>${c.separation}</td><td>${kw}</td></tr>`;
  }
  h += '</tbody></table>';
  $('clusterTable').innerHTML = h;
  $('clusterTable').querySelectorAll('tr[data-cid]').forEach(tr => {
    tr.addEventListener('click', () => {
      document.querySelectorAll('#clusterTable tr').forEach(r => r.classList.remove('highlighted-row'));
      tr.classList.add('highlighted-row');
      openMembers(parseInt(tr.dataset.cid));
    });
  });
}

let DRAWER_CID = null;
async function openMembers(cid){
  if(!CURRENT_RUN_ID) return;
  DRAWER_CID = cid;

  const drawer = $('drawer');
  drawer.classList.add('open');
  document.body.classList.add('drawer-open');
  const widthVal = document.body.style.getPropertyValue('--drawer-width') || '440px';
  document.body.style.setProperty('--drawer-width', widthVal);
  setTimeout(triggerPlotResize, 50);

  $('drawerTitle').textContent = `Cluster ${cid} members`;
  $('drawerBody').innerHTML = 'Loading…';
  try {
    const r = await api(`/api/clusters/${encodeURIComponent(CURRENT_RUN_ID)}/members?cluster_id=${cid}&limit=200`);
    let h = `<p class="hint">${r.total} members</p>`;
    for(const m of r.members){
      h += `<div class="member"><div class="mid">${esc(m.id)} <span class="prob">p=${m.probability} · sim=${m.self_similarity}</span></div>${esc(m.text)}</div>`;
    }
    $('drawerBody').innerHTML = h;
  } catch(e){ $('drawerBody').innerHTML = 'Error: ' + esc(e.message); }
}

async function centroidSearch(){
  if(!CURRENT_RUN_ID || DRAWER_CID == null) return;
  $('drawerBody').innerHTML = 'Searching by centroid…';
  try {
    const r = await api(`/api/clusters/${encodeURIComponent(CURRENT_RUN_ID)}/search`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ cluster_id: DRAWER_CID, top_k: 20 }),
    });
    let h = `<p class="hint">Top ${r.chunks.length} by centroid similarity</p>`;
    for(const c of r.chunks){
      h += `<div class="member"><div class="mid">${esc(c.chunk_id)} <span class="prob">score=${(c.score||0).toFixed(4)}</span></div>${esc(c.text||'')}</div>`;
    }
    $('drawerBody').innerHTML = h;
  } catch(e){ $('drawerBody').innerHTML = 'Error: ' + esc(e.message); }
}

function updateControlsVisibility(){
  const algo = $('algorithm').value;
  const ids = [
    'ctrl_min_cluster_size', 'ctrl_k', 'ctrl_eps', 'ctrl_min_samples',
    'ctrl_selection_method', 'ctrl_linkage', 'ctrl_distance_threshold',
    'ctrl_resolution', 'ctrl_knn'
  ];

  // By default, hide all algorithm-specific controls
  ids.forEach(id => { const el = $(id); if(el) el.style.display = 'none'; });

  // Show only relevant controls based on chosen algorithm
  if(algo === 'hdbscan'){
    $('ctrl_min_cluster_size').style.display = '';
    $('ctrl_min_samples').style.display = '';
    $('ctrl_selection_method').style.display = '';
  } else if(algo === 'dbscan'){
    $('ctrl_eps').style.display = '';
    $('ctrl_min_samples').style.display = '';
  } else if(algo === 'kmeans'){
    $('ctrl_k').style.display = '';
  } else if(algo === 'agglomerative'){
    $('ctrl_k').style.display = '';
    $('ctrl_linkage').style.display = '';
    $('ctrl_distance_threshold').style.display = '';
  } else if(algo === 'leiden'){
    $('ctrl_resolution').style.display = '';
    $('ctrl_knn').style.display = '';
  } else if(algo === 'auto'){
    $('ctrl_min_cluster_size').style.display = '';
    $('ctrl_k').style.display = '';
  }
}

$('runBtn').addEventListener('click', runClustering);
$('loadRunBtn').addEventListener('click', loadRun);
$('drawerClose').addEventListener('click', () => {
  $('drawer').classList.remove('open');
  document.body.classList.remove('drawer-open');
  setTimeout(triggerPlotResize, 200);
});
$('centroidSearchBtn').addEventListener('click', centroidSearch);
$('algorithm').addEventListener('change', updateControlsVisibility);
loadVersion();
refreshRuns();
updateControlsVisibility();

function triggerPlotResize() {
  try {
    if($('sizes').children.length) Plotly.Plots.resize('sizes');
    if($('aux').children.length) Plotly.Plots.resize('aux');
    if($('scatter').children.length) Plotly.Plots.resize('scatter');
  } catch(e) {}
}

// Buttery-smooth Plotly responsive resizing via ResizeObserver
let resizeTimeout;
const ro = new ResizeObserver(() => {
  cancelAnimationFrame(resizeTimeout);
  resizeTimeout = requestAnimationFrame(triggerPlotResize);
});
const sizesPanel = $('sizesPanel');
const auxPanel = $('auxPanel');
if(sizesPanel) ro.observe(sizesPanel);
if(auxPanel) ro.observe(auxPanel);

// Dynamic Drawer resizing logic
(function() {
  const drawer = $('drawer');
  if (!drawer) return;

  const resizer = document.createElement('div');
  resizer.className = 'drawer-resizer';
  resizer.id = 'drawerResizer';
  drawer.insertBefore(resizer, drawer.firstChild);

  let isDraggingDrawer = false;
  let startWidth, startX;

  resizer.addEventListener('mousedown', (e) => {
    isDraggingDrawer = true;
    resizer.classList.add('dragging');
    startWidth = drawer.offsetWidth;
    startX = e.clientX;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDraggingDrawer) return;
    const deltaX = startX - e.clientX;
    const newWidth = Math.max(280, Math.min(window.innerWidth - 100, startWidth + deltaX));
    document.body.style.setProperty('--drawer-width', `${newWidth}px`);
    triggerPlotResize();
  });

  document.addEventListener('mouseup', () => {
    if (isDraggingDrawer) {
      isDraggingDrawer = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      triggerPlotResize();
    }
  });
})();
