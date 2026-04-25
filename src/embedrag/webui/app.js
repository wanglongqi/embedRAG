/* EmbedRAG WebUI -- vanilla JS, no build step */
(function () {
  "use strict";

  const API_BASE = window.location.origin;

  // ── DOM refs ──
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // Theme
  const themeBtn = $("#themeToggle");
  const root = document.documentElement;

  function initTheme() {
    const saved = localStorage.getItem("embedrag-theme");
    if (saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      root.setAttribute("data-theme", "dark");
    }
  }
  initTheme();

  themeBtn.addEventListener("click", () => {
    const isDark = root.getAttribute("data-theme") === "dark";
    root.setAttribute("data-theme", isDark ? "light" : "dark");
    localStorage.setItem("embedrag-theme", isDark ? "light" : "dark");
  });

  // ── Tab navigation ──
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      $$(".tab-content").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "status") refreshStatus();
    });
  });

  // ── Sliders ──
  $("#topK").addEventListener("input", (e) => { $("#topKValue").textContent = e.target.value; });
  $("#debugTopK").addEventListener("input", (e) => { $("#debugTopKValue").textContent = e.target.value; });

  // ── Helpers ──
  function buildFilters() {
    const filters = {};
    const dt = $("#filterDocType").value.trim();
    const di = $("#filterDocId").value.trim();
    if (dt) filters.doc_type = dt;
    if (di) filters.doc_id = di;
    return Object.keys(filters).length > 0 ? filters : null;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function truncate(s, n) {
    return s.length > n ? s.slice(0, n) + "..." : s;
  }

  async function apiFetch(path, body) {
    const resp = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text}`);
    }
    return resp.json();
  }

  async function apiGet(path) {
    const resp = await fetch(API_BASE + path);
    if (!resp.ok) throw new Error(`${resp.status}`);
    return resp.json();
  }

  // ── Neighbor context viewer ──
  async function loadNeighbors(chunkId, container, before, after) {
    const existingChunks = container.querySelector(".neighbor-chunks");
    if (!existingChunks) {
      container.innerHTML = '<div class="empty-state"><span class="spinner"></span> Loading...</div>';
    }

    try {
      const data = await apiGet(`/api/chunks/${encodeURIComponent(chunkId)}/neighbors?before=${before}&after=${after}`);

      let chunksHtml = "";
      const hasBefore = data.before && data.before.length > 0;
      const hasAfter = data.after && data.after.length > 0;
      const moreBefore = data.before && data.before.length >= before;
      const moreAfter = data.after && data.after.length >= after;

      if (moreBefore) {
        chunksHtml += `<div class="neighbor-controls neighbor-controls-top"><button class="btn-sm" data-action="more-before">Load more above</button></div>`;
      }

      if (hasBefore) {
        for (const nb of data.before) {
          chunksHtml += `<div class="neighbor-chunk neighbor-before">${escapeHtml(nb.text)}</div>`;
        }
      }

      chunksHtml += `<div class="neighbor-chunk neighbor-current">${escapeHtml(data.current ? data.current.text : "")}</div>`;

      if (hasAfter) {
        for (const nb of data.after) {
          chunksHtml += `<div class="neighbor-chunk neighbor-after">${escapeHtml(nb.text)}</div>`;
        }
      }

      if (moreAfter) {
        chunksHtml += `<div class="neighbor-controls neighbor-controls-bottom"><button class="btn-sm" data-action="more-after">Load more below</button></div>`;
      }

      container.innerHTML = `<div class="neighbor-chunks">${chunksHtml}</div>`;

      // Scroll current chunk into view within the container
      const cur = container.querySelector(".neighbor-current");
      if (cur) cur.scrollIntoView({ block: "center", behavior: "instant" });

      // Bind load-more buttons
      container.querySelectorAll("[data-action='more-before']").forEach((btn) => {
        btn.addEventListener("click", () => {
          const v = btn.closest(".neighbor-viewer");
          v.dataset.before = parseInt(v.dataset.before || "3") + 5;
          loadNeighbors(chunkId, v, parseInt(v.dataset.before), parseInt(v.dataset.after || "3"));
        });
      });
      container.querySelectorAll("[data-action='more-after']").forEach((btn) => {
        btn.addEventListener("click", () => {
          const v = btn.closest(".neighbor-viewer");
          v.dataset.after = parseInt(v.dataset.after || "3") + 5;
          loadNeighbors(chunkId, v, parseInt(v.dataset.before || "3"), parseInt(v.dataset.after));
        });
      });
    } catch (err) {
      container.innerHTML = `<div class="empty-state" style="color:var(--error)">Error: ${escapeHtml(err.message)}</div>`;
    }
  }

  window.__loadNb = function (chunkId, viewer) {
    const b = parseInt(viewer.dataset.before || "3");
    const a = parseInt(viewer.dataset.after || "3");
    loadNeighbors(chunkId, viewer, b, a);
  };

  // ── Reranker config (persisted in localStorage) ──
  const RERANK_DEFAULTS = {
    url: "http://127.0.0.1:1234/v1/embeddings",
    model: "text-embedding-bge-reranker-v2-m3",
  };
  const RERANK_TEXT_LIMIT = 512;

  function getRerankConfig() {
    return {
      url: localStorage.getItem("embedrag-rerank-url") || RERANK_DEFAULTS.url,
      model: localStorage.getItem("embedrag-rerank-model") || RERANK_DEFAULTS.model,
    };
  }

  function initRerankInputs() {
    const cfg = getRerankConfig();
    const urlInput = $("#rerankUrl");
    const modelInput = $("#rerankModel");
    urlInput.value = cfg.url;
    modelInput.value = cfg.model;
    urlInput.addEventListener("change", () => {
      const v = urlInput.value.trim();
      if (v) localStorage.setItem("embedrag-rerank-url", v);
      else localStorage.removeItem("embedrag-rerank-url");
    });
    modelInput.addEventListener("change", () => {
      const v = modelInput.value.trim();
      if (v) localStorage.setItem("embedrag-rerank-model", v);
      else localStorage.removeItem("embedrag-rerank-model");
    });
  }
  initRerankInputs();

  let _lastSearchQuery = "";
  let _lastSearchChunks = [];
  let _lastScoreType = "";

  async function callRerank(query, chunks, scoreType) {
    const cfg = getRerankConfig();
    const texts = chunks.map((c) => (c.text || "").slice(0, RERANK_TEXT_LIMIT));
    const data = await apiFetch("/api/rerank", {
      query, texts, url: cfg.url, model: cfg.model,
    });
    const byIndex = new Map(data.results.map((r) => [r.index, r.score]));
    const scored = chunks.map((c, i) => ({
      ...c,
      _origScore: c._origScore !== undefined ? c._origScore : c.score,
      _origScoreType: c._origScoreType || scoreType,
      score: byIndex.get(i) ?? 0,
    }));
    scored.sort((a, b) => b.score - a.score);
    return { scored, elapsed_ms: data.elapsed_ms };
  }

  async function doRerank() {
    if (!_lastSearchChunks.length) return;

    const btn = $("#rerankBtn");
    const status = $("#rerankStatus");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    status.textContent = "";
    status.style.color = "";

    try {
      const { scored, elapsed_ms } = await callRerank(
        _lastSearchQuery, _lastSearchChunks, _lastScoreType
      );
      _lastSearchChunks = scored;

      const container = $("#searchResults");
      container.innerHTML = "";
      const hint = document.createElement("div");
      hint.className = "score-hint";
      hint.textContent = "Sorted by Rerank score. Original score shown alongside.";
      container.appendChild(hint);
      scored.forEach((c) => container.appendChild(renderChunkCard(c, "rerank")));

      const timingBar = $("#searchTiming");
      if (!timingBar.querySelector(".timing-rerank")) {
        const chip = document.createElement("span");
        chip.className = "timing-chip timing-rerank";
        chip.textContent = `Rerank: ${elapsed_ms.toFixed(0)}ms`;
        timingBar.appendChild(chip);
      }

      btn.textContent = "Reranked";
      btn.classList.add("reranked");
      status.textContent = `${scored.length} results, ${elapsed_ms.toFixed(0)}ms`;
    } catch (err) {
      status.textContent = err.message;
      status.style.color = "var(--error)";
      btn.textContent = "Rerank";
      btn.disabled = false;
    }
  }

  // ── Score formatting ──
  const SCORE_LABELS = {
    rrf: "RRF",
    cosine: "Sim",
    bm25: "BM25",
    rerank: "Rerank",
  };

  function formatScoreWithType(score, scoreType) {
    const label = SCORE_LABELS[scoreType] || scoreType || "";
    const s = Math.abs(score);
    let display;
    if (s === 0) {
      display = "0";
    } else if (scoreType === "rrf") {
      display = (s * 1000).toFixed(1);
    } else if (scoreType === "cosine" || scoreType === "rerank") {
      display = s.toFixed(3);
    } else if (s >= 100) {
      display = s.toFixed(0);
    } else if (s >= 1) {
      display = s.toFixed(2);
    } else {
      display = s.toFixed(4);
    }
    return { display, label };
  }

  // ── Render: chunk result card ──
  function renderChunkCard(chunk, scoreType) {
    const card = document.createElement("div");
    card.className = "result-card";
    const m = chunk.metadata || {};

    const title = m.title || "";
    const author = m.author || "";
    const docType = m.doc_type || chunk.doc_type || "";

    const meta = [
      chunk.chunk_id ? `<span class="badge badge-chunkid" title="${escapeHtml(chunk.chunk_id)}"><code>${escapeHtml(truncate(chunk.chunk_id, 16))}</code></span>` : "",
      title ? `<span class="badge badge-title">${escapeHtml(title)}</span>` : "",
      author ? `<span class="badge badge-author">${escapeHtml(author)}</span>` : "",
      docType ? `<span class="badge badge-doctype">${escapeHtml(docType)}</span>` : "",
      `<span class="badge">doc: ${escapeHtml(truncate(chunk.doc_id || "", 30))}</span>`,
      `<span class="badge level-type">${escapeHtml(chunk.level_type || "chunk")}</span>`,
      chunk.level !== undefined ? `<span class="badge">L${chunk.level}</span>` : "",
    ].filter(Boolean);

    let parentHtml = "";
    if (chunk.parent_text) {
      const sections = chunk.parent_text.split("\n---\n");
      const body = sections.map((s) => `<div class="parent-section">${escapeHtml(s)}</div>`).join("");
      parentHtml = `
        <details class="result-parent">
          <summary>Hierarchy (${sections.length} level${sections.length > 1 ? "s" : ""})</summary>
          ${body}
        </details>`;
    }

    const neighborId = `nb-${chunk.chunk_id.replace(/[^a-zA-Z0-9]/g, "")}`;

    const mainScore = formatScoreWithType(chunk.score, scoreType);
    let scoreHtml = `<span class="result-score" title="Raw: ${chunk.score}">${mainScore.label ? mainScore.label + " " + mainScore.display : mainScore.display}</span>`;
    if (chunk._origScore !== undefined && chunk._origScoreType) {
      const orig = formatScoreWithType(chunk._origScore, chunk._origScoreType);
      scoreHtml = `<span class="result-score score-rerank" title="Rerank raw: ${chunk.score}">${mainScore.label} ${mainScore.display}</span>`
        + `<span class="result-score score-orig" title="Original raw: ${chunk._origScore}">${orig.label} ${orig.display}</span>`;
    }

    card.innerHTML = `
      <div class="result-header">
        <div class="result-meta">${meta.join("")}</div>
        <div class="result-scores">${scoreHtml}</div>
      </div>
      <div class="result-text">${escapeHtml(chunk.text || "")}</div>
      ${parentHtml}
      <details class="result-neighbors" ontoggle="if(this.open && !this.dataset.loaded){this.dataset.loaded='1';window.__loadNb('${chunk.chunk_id}',this.querySelector('.neighbor-viewer'))}">
        <summary>Surrounding context</summary>
        <div class="neighbor-viewer" id="${neighborId}" data-before="3" data-after="3"></div>
      </details>`;
    return card;
  }

  // ── Render: timing chips ──
  function renderTimingChips(container, data) {
    container.style.display = "flex";
    const items = [
      ["Embed", data.embedding_time_ms || data.embedding_ms || 0],
      ["Dense", data.dense_time_ms || data.dense_ms || 0],
      ["Sparse", data.sparse_time_ms || data.sparse_ms || 0],
      ["Fusion", data.fusion_time_ms || data.fusion_ms || 0],
      ["Total", data.total_time_ms || data.total_ms || 0],
    ];
    container.innerHTML = items
      .map(([label, ms]) => {
        const cls = label === "Total" ? "timing-chip total" : "timing-chip";
        return `<span class="${cls}">${label}: ${ms.toFixed(1)}ms</span>`;
      })
      .join("");
  }

  // ── Search ──
  function resetRerankBtn(hasResults) {
    const btn = $("#rerankBtn");
    const status = $("#rerankStatus");
    if (hasResults) {
      btn.style.display = "";
      btn.textContent = "Rerank";
      btn.disabled = false;
      btn.classList.remove("reranked");
      status.textContent = "";
      status.style.color = "";
    } else {
      btn.style.display = "none";
      status.textContent = "";
    }
  }

  async function doSearch() {
    const query = $("#searchQuery").value.trim();
    if (!query) return;

    const btn = $("#searchBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    $("#searchResults").innerHTML = "";
    $("#searchTiming").style.display = "none";
    resetRerankBtn(false);
    _lastSearchChunks = [];
    _lastSearchQuery = "";
    _lastScoreType = "";

    try {
      const data = await apiFetch("/search/text", {
        query_text: query,
        top_k: parseInt($("#topK").value),
        mode: $("#searchMode").value,
        space: $("#searchSpace").value,
        filters: buildFilters(),
        expand_context: true,
        context_depth: 1,
      });

      renderTimingChips($("#searchTiming"), data);

      const container = $("#searchResults");
      if (!data.chunks || data.chunks.length === 0) {
        container.innerHTML = '<div class="empty-state">No results found.</div>';
        return;
      }

      _lastSearchQuery = query;
      _lastSearchChunks = data.chunks.slice();
      _lastScoreType = data.score_type || "";

      const scoreType = _lastScoreType;
      const scaleHints = {
        rrf: "RRF score ×1000 (rank-based, higher = better)",
        cosine: "Cosine similarity (0-1, higher = closer)",
        bm25: "BM25 text relevance (higher = better match)",
      };
      if (scaleHints[scoreType]) {
        const hint = document.createElement("div");
        hint.className = "score-hint";
        hint.textContent = scaleHints[scoreType];
        container.appendChild(hint);
      }
      data.chunks.forEach((c) => container.appendChild(renderChunkCard(c, scoreType)));
      resetRerankBtn(true);
    } catch (err) {
      $("#searchResults").innerHTML = `<div class="empty-state" style="color:var(--error)">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Search";
    }
  }

  $("#searchBtn").addEventListener("click", doSearch);
  $("#searchQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  $("#rerankBtn").addEventListener("click", doRerank);

  // ── Debug ──
  let _debugQuery = "";
  let _debugChunks = [];
  let _debugScoreType = "";

  function renderDebugTable(container, rows, columns) {
    if (!rows || rows.length === 0) {
      container.innerHTML = '<div class="empty-state">No results</div>';
      return;
    }
    const ths = columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("");
    const trs = rows
      .map((r) => {
        const tds = columns.map((c) => `<td>${typeof c.render === "function" ? c.render(r) : escapeHtml(String(r[c.key] ?? ""))}</td>`).join("");
        return `<tr>${tds}</tr>`;
      })
      .join("");
    container.innerHTML = `<table class="debug-table"><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
  }

  function renderWaterfall(container, timing) {
    const stages = [
      { key: "embedding_ms", label: "Embed", cls: "embed" },
      { key: "dense_ms", label: "Dense", cls: "dense" },
      { key: "sparse_ms", label: "Sparse", cls: "sparse" },
      { key: "fusion_ms", label: "Fusion", cls: "fusion" },
      { key: "fetch_ms", label: "Fetch", cls: "fetch" },
      { key: "expand_ms", label: "Expand", cls: "expand" },
    ];
    const maxMs = Math.max(timing.total_ms || 1, ...stages.map((s) => timing[s.key] || 0));

    container.innerHTML = stages
      .map((s) => {
        const ms = timing[s.key] || 0;
        const pct = Math.max((ms / maxMs) * 100, ms > 0 ? 2 : 0);
        return `
          <div class="waterfall-row">
            <span class="waterfall-label">${s.label}</span>
            <div class="waterfall-bar-bg">
              <div class="waterfall-bar ${s.cls}" style="width:${pct}%">${ms > 5 ? ms.toFixed(1) : ""}</div>
            </div>
            <span class="waterfall-ms">${ms.toFixed(1)}ms</span>
          </div>`;
      })
      .join("");
  }

  function resetDebugRerank(hasResults) {
    const btn = $("#debugRerankBtn");
    const status = $("#debugRerankStatus");
    if (hasResults) {
      btn.style.display = "";
      btn.textContent = "Rerank";
      btn.disabled = false;
      btn.classList.remove("reranked");
      status.textContent = "";
      status.style.color = "";
    } else {
      btn.style.display = "none";
      status.textContent = "";
    }
    $("#debugRerankSection").style.display = "none";
  }

  async function doDebug() {
    const query = $("#debugQuery").value.trim();
    if (!query) return;

    const btn = $("#debugBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    $("#debugOutput").style.display = "none";
    resetDebugRerank(false);
    _debugQuery = "";
    _debugChunks = [];
    _debugScoreType = "";

    try {
      const data = await apiFetch("/api/debug/search", {
        query_text: query,
        top_k: parseInt($("#debugTopK").value),
        mode: $("#debugMode").value,
        expand_context: true,
        context_depth: 1,
      });

      $("#debugOutput").style.display = "block";
      $("#debugConfig").textContent = JSON.stringify(data.config_snapshot, null, 2);
      $("#debugFtsQuery").textContent = data.fts_query || "(none)";

      renderWaterfall($("#debugWaterfall"), data.timing);

      // Dense
      $("#denseCount").textContent = data.dense_results ? data.dense_results.length : 0;
      renderDebugTable($("#debugDense"), data.dense_results, [
        { key: "chunk_id", label: "Chunk ID", render: (r) => `<code>${escapeHtml(truncate(r.chunk_id, 24))}</code>` },
        { key: "score", label: "Score", render: (r) => r.score.toFixed(6) },
      ]);

      // Sparse
      $("#sparseCount").textContent = data.sparse_results ? data.sparse_results.length : 0;
      renderDebugTable($("#debugSparse"), data.sparse_results, [
        { key: "chunk_id", label: "Chunk ID", render: (r) => `<code>${escapeHtml(truncate(r.chunk_id, 24))}</code>` },
        { key: "score", label: "BM25 Score", render: (r) => r.score.toFixed(6) },
      ]);

      // Fused
      $("#fusedCount").textContent = data.fused_results ? data.fused_results.length : 0;
      renderDebugTable($("#debugFused"), data.fused_results, [
        { key: "chunk_id", label: "Chunk ID", render: (r) => `<code>${escapeHtml(truncate(r.chunk_id, 24))}</code>` },
        { key: "rrf_score", label: "RRF Score", render: (r) => r.rrf_score.toFixed(6) },
        { key: "dense_score", label: "Dense", render: (r) => r.dense_score.toFixed(4) },
        { key: "sparse_score", label: "Sparse", render: (r) => r.sparse_score.toFixed(4) },
        { key: "dense_rank", label: "D.Rank" },
        { key: "sparse_rank", label: "S.Rank" },
      ]);

      // Final chunks
      _debugQuery = query;
      _debugScoreType = data.score_type || (data.fused_results && data.fused_results.length > 0 ? "rrf" : "cosine");
      const chunksContainer = $("#debugChunks");
      chunksContainer.innerHTML = "";
      if (data.final_chunks && data.final_chunks.length > 0) {
        _debugChunks = data.final_chunks.slice();
        data.final_chunks.forEach((c) => chunksContainer.appendChild(renderChunkCard(c, _debugScoreType)));
        resetDebugRerank(true);
      } else {
        _debugChunks = [];
        chunksContainer.innerHTML = '<div class="empty-state">No final chunks.</div>';
      }
    } catch (err) {
      $("#debugOutput").style.display = "block";
      $("#debugOutput").innerHTML = `<div class="empty-state" style="color:var(--error)">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Debug Search";
    }
  }

  async function doDebugRerank() {
    if (!_debugChunks.length) return;

    const btn = $("#debugRerankBtn");
    const status = $("#debugRerankStatus");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    status.textContent = "";
    status.style.color = "";

    try {
      const origOrder = _debugChunks.map((c) => c.chunk_id);
      const { scored, elapsed_ms } = await callRerank(
        _debugQuery, _debugChunks, _debugScoreType
      );
      _debugChunks = scored;

      const comparison = scored.map((c, newRank) => {
        const origRank = origOrder.indexOf(c.chunk_id);
        return {
          chunk_id: c.chunk_id,
          title: (c.metadata || {}).title || c.doc_id || "",
          orig_rank: (origRank >= 0 ? origRank : newRank) + 1,
          new_rank: newRank + 1,
          rank_delta: ((origRank >= 0 ? origRank : newRank) + 1) - (newRank + 1),
          orig_score: c._origScore,
          orig_type: c._origScoreType,
          rerank_score: c.score,
        };
      });

      const section = $("#debugRerankSection");
      section.style.display = "block";
      renderDebugTable($("#debugRerankTable"), comparison, [
        { key: "new_rank", label: "#" },
        { key: "chunk_id", label: "Chunk ID", render: (r) => `<code>${escapeHtml(truncate(r.chunk_id, 16))}</code>` },
        { key: "title", label: "Title", render: (r) => escapeHtml(truncate(r.title, 30)) },
        { key: "orig_score", label: "Orig Score", render: (r) => {
          const f = formatScoreWithType(r.orig_score, r.orig_type);
          return `${f.label} ${f.display}`;
        }},
        { key: "rerank_score", label: "Rerank", render: (r) => r.rerank_score.toFixed(3) },
        { key: "rank_delta", label: "Move", render: (r) => {
          if (r.rank_delta > 0) return `<span style="color:var(--success)">+${r.rank_delta}</span>`;
          if (r.rank_delta < 0) return `<span style="color:var(--error)">${r.rank_delta}</span>`;
          return `<span style="color:var(--text-secondary)">=</span>`;
        }},
      ]);

      // Re-render final chunks with dual scores
      const chunksContainer = $("#debugChunks");
      chunksContainer.innerHTML = "";
      scored.forEach((c) => chunksContainer.appendChild(renderChunkCard(c, "rerank")));

      btn.textContent = "Reranked";
      btn.classList.add("reranked");
      status.textContent = `${scored.length} results, ${elapsed_ms.toFixed(0)}ms`;
    } catch (err) {
      status.textContent = err.message;
      status.style.color = "var(--error)";
      btn.textContent = "Rerank";
      btn.disabled = false;
    }
  }

  $("#debugBtn").addEventListener("click", doDebug);
  $("#debugQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") doDebug(); });
  $("#debugRerankBtn").addEventListener("click", doDebugRerank);

  // ── Status ──
  function _tsToRelative(ts) {
    if (!ts) return "--";
    const ago = (Date.now() / 1000) - ts;
    if (ago < 0) return new Date(ts * 1000).toLocaleTimeString();
    if (ago < 60) return `${Math.round(ago)}s ago`;
    if (ago < 3600) return `${Math.round(ago / 60)}m ago`;
    if (ago < 86400) return `${(ago / 3600).toFixed(1)}h ago`;
    return new Date(ts * 1000).toLocaleDateString();
  }

  function _tsToEta(ts) {
    if (!ts) return "--";
    const left = ts - (Date.now() / 1000);
    if (left <= 0) return "any moment";
    if (left < 60) return `in ${Math.round(left)}s`;
    if (left < 3600) return `in ${Math.round(left / 60)}m`;
    return `in ${(left / 3600).toFixed(1)}h`;
  }

  const SYNC_RESULT_LABELS = {
    none: { text: "Never synced", cls: "muted" },
    up_to_date: { text: "Up to date", cls: "ok" },
    synced: { text: "Synced", cls: "ok" },
    no_latest: { text: "No latest.json", cls: "warn" },
    verify_failed: { text: "Verify failed", cls: "error" },
    error: { text: "Error", cls: "error" },
  };

  async function refreshSyncStatus() {
    const setSyncVal = (id, text, cls) => {
      const el = $(`#${id}`);
      el.textContent = text;
      el.className = `sync-val ${cls || ""}`;
    };

    try {
      const s = await apiGet("/admin/sync/status");

      setSyncVal("syncEnabled", s.enabled ? "Enabled" : "Disabled", s.enabled ? "ok" : "muted");
      setSyncVal("syncSource", s.source || "--");

      const schedule = s.cron || (s.poll_interval_seconds ? `every ${s.poll_interval_seconds}s` : "--");
      setSyncVal("syncSchedule", s.enabled ? schedule : "--", s.enabled ? "" : "muted");

      setSyncVal("syncLastCheck", _tsToRelative(s.last_check_at));

      const rl = SYNC_RESULT_LABELS[s.last_result] || { text: s.last_result, cls: "" };
      setSyncVal("syncLastResult", rl.text, rl.cls);

      setSyncVal("syncLastVersion", s.last_version || "--", s.last_version ? "" : "muted");
      setSyncVal("syncNextCheck", s.enabled && s.next_check_at ? _tsToEta(s.next_check_at) : "--", s.enabled ? "" : "muted");
      setSyncVal("syncErrors", String(s.consecutive_errors), s.consecutive_errors > 0 ? "error" : "ok");
    } catch {
      setSyncVal("syncEnabled", "unavailable", "error");
    }
  }

  async function refreshStatus() {
    const setVal = (id, text, cls) => {
      const el = $(`#${id}`);
      el.textContent = text;
      el.className = `status-value ${cls || ""}`;
    };

    try {
      const health = await apiGet("/health");
      setVal("statusHealth", health.status || "ok", health.status === "ok" ? "ok" : "error");
    } catch {
      setVal("statusHealth", "unreachable", "error");
    }

    try {
      const ready = await apiGet("/readiness");
      setVal("statusVersion", ready.active_version || "--");
      setVal("statusVectors", ready.vector_count ? ready.vector_count.toLocaleString() : "0");
      setVal("statusDocs", ready.doc_count ? ready.doc_count.toLocaleString() : "0");
      $("#versionBadge").textContent = ready.active_version || "--";
    } catch {
      setVal("statusVersion", "--", "unknown");
      setVal("statusVectors", "--", "unknown");
      setVal("statusDocs", "--", "unknown");
    }

    try {
      await apiFetch("/search/text", { query_text: "ping", top_k: 1, mode: "dense" });
      setVal("statusEmbed", "connected", "ok");
    } catch {
      setVal("statusEmbed", "unavailable", "error");
    }

    await refreshSyncStatus();
  }

  async function doSyncTrigger() {
    const btn = $("#syncTriggerBtn");
    const status = $("#syncTriggerStatus");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    status.textContent = "";
    status.style.color = "";
    try {
      const data = await apiFetch("/admin/sync", {});
      status.textContent = data.status || "done";
      status.style.color = data.status === "synced" ? "var(--success)" : "";
      await refreshSyncStatus();
    } catch (err) {
      status.textContent = err.message;
      status.style.color = "var(--error)";
    } finally {
      btn.disabled = false;
      btn.textContent = "Sync Now";
    }
  }

  async function _doSyncAction(body, btn, statusEl, defaultLabel) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    statusEl.textContent = "";
    statusEl.style.color = "";
    try {
      const data = await apiFetch("/admin/sync", body);
      const detail = data.version ? ` (${data.version})` : (data.source ? ` (${data.source})` : "");
      statusEl.textContent = data.status + detail;
      statusEl.style.color = data.status === "synced" ? "var(--success)" : "";
      await refreshSyncStatus();
      await refreshStatus();
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.style.color = "var(--error)";
    } finally {
      btn.disabled = false;
      btn.textContent = defaultLabel;
    }
  }

  function doSyncFromUrl() {
    const url = $("#syncUrlInput").value.trim();
    if (!url) return;
    _doSyncAction({ source_url: url }, $("#syncUrlBtn"), $("#syncUrlStatus"), "Sync");
  }

  function doSyncFromDir() {
    const dir = $("#syncDirInput").value.trim();
    if (!dir) return;
    _doSyncAction({ snapshot_dir: dir }, $("#syncDirBtn"), $("#syncDirStatus"), "Load");
  }

  $("#statusRefresh").addEventListener("click", refreshStatus);
  $("#syncTriggerBtn").addEventListener("click", doSyncTrigger);
  $("#syncUrlBtn").addEventListener("click", doSyncFromUrl);
  $("#syncUrlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doSyncFromUrl(); });
  $("#syncDirBtn").addEventListener("click", doSyncFromDir);
  $("#syncDirInput").addEventListener("keydown", (e) => { if (e.key === "Enter") doSyncFromDir(); });

  // ── Space selector population ──
  async function loadSpaces() {
    try {
      const data = await apiGet("/api/spaces");
      const spaces = data.spaces || ["text"];
      ["searchSpace", "debugSpace"].forEach((id) => {
        const sel = $(`#${id}`);
        sel.innerHTML = spaces.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join("");
      });
    } catch { /* keep default "text" */ }
  }

  // Load version badge + spaces on startup
  Promise.all([
    apiGet("/readiness")
      .then((d) => { $("#versionBadge").textContent = d.active_version || "--"; })
      .catch(() => { $("#versionBadge").textContent = "--"; }),
    loadSpaces(),
  ]);
})();
