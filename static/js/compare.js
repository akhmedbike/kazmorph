/**
 * KazMorph — A/B comparison page (/compare).
 *
 * Calls both /api/v1/correct (binary) and /api/v1/correct3 (three-way) with
 * the same text and renders the results side by side. The three-way panel
 * color-codes each token by its decision (KEEP / CORRECT / REVIEW) and shows
 * the candidate list under REVIEW tokens.
 *
 * Self-contained — does not depend on app.js / i18n.js (the compare page is a
 * standalone tool). Errors are surfaced inline rather than thrown.
 */
(function () {
  "use strict";

  const SAMPLES = [
    "Бүгін адамм мектепке келді және жаңа мұғаліммен танысты.",
    "Қазақстан Республикасының астанасы Астана қаласы.",
    "оқушыларымыздың жетістіктері үшін мерекелік концерт ұйымдастырдық.",
  ];
  let sampleIdx = 0;

  function esc(s) {
    // escape user text before injecting into innerHTML
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ── Render the binary /correct result ────────────────────────────────────
  // The binary endpoint returns { corrected, corrections:[{token,suggestion,…}] }.
  // We walk the original text and wrap any token that appears in `corrections`
  // (matched by position) in a CORRECT-colored span so the user sees WHAT changed.
  function renderOld(data, text) {
    const meta = document.getElementById("cmpOld");
    const metaBar = document.getElementById("cmpOldMeta");
    if (!data || data.error) {
      meta.innerHTML = '<span class="ph">қате: ' + esc((data && data.error) || "—") + "</span>";
      metaBar.textContent = "";
      return { changed: 0 };
    }
    // Build a position->suggestion map for highlighting.
    const byPos = {};
    (data.corrections || []).forEach((c) => { byPos[c.position] = c; });
    // Reconstruct with highlights using the same approach as the server: scan
    // the text and wrap tokens whose start position is in byPos.
    let html = "";
    let i = 0;
    while (i < text.length) {
      const c = byPos[i];
      if (c) {
        // this is a corrected token — find its end (run of non-space)
        let j = i;
        while (j < text.length && !/\s/.test(text[j])) j++;
        html += '<span class="tok correct" title="түзетілді: ' + esc(c.token) +
                " → " + esc(c.suggestion) + '">' + esc(c.suggestion) + "</span>";
        i = j;
      } else {
        // copy whitespace/punctuation runs verbatim until the next token
        let j = i;
        while (j < text.length && (/\s/.test(text[j]) || !(byPos[j]))) {
          // emit one char at a time but batch contiguous non-token chars
          if (byPos[j]) break;
          html += esc(text[j]);
          j++;
        }
        if (j === i) j++; // safety: always advance
        i = j;
      }
    }
    meta.innerHTML = '<div class="cmp-out">' + html + "</div>";
    const st = data.stats || {};
    metaBar.textContent = "токендер: " + (st.tokens || "?") +
                          " · түзетілді: " + (st.corrected || 0);
    return { changed: st.corrected || 0 };
  }

  // ── Render the three-way /correct3 result ────────────────────────────────
  // The three-way endpoint returns { corrected, decisions:[{token,predicted,
  // decision, top_candidate, candidates:[…]}], stats:{keep,correct,review} }.
  // We render the corrected text with per-token decision coloring and attach
  // the candidate list under REVIEW tokens.
  function renderNew(data, text) {
    const meta = document.getElementById("cmpNew");
    const metaBar = document.getElementById("cmpNewMeta");
    if (!data || data.error) {
      meta.innerHTML = '<span class="ph">қате: ' + esc((data && data.error) || "—") + "</span>";
      metaBar.textContent = "";
      return { changed: 0, review: 0 };
    }
    const decisions = data.decisions || [];
    // Reconstruct from the decision list: emitted surface forms in order,
    // with inter-token whitespace copied from the original text.
    // The server already computed `corrected`; but to color tokens we rebuild.
    let html = "";
    let cursor = 0;
    decisions.forEach((d) => {
      // find the token's original span in text starting at cursor
      const idx = text.indexOf(d.token, cursor);
      if (idx === -1) { return; }
      // copy any whitespace/punct between cursor and the token
      if (idx > cursor) html += esc(text.slice(cursor, idx));
      const cls = "tok " + (d.decision || "keep").toLowerCase();
      const emitted = d.decision === "CORRECT" ? d.predicted : d.token;
      let title = d.decision + " · p_err=" + (d.p_err != null ? d.p_err.toFixed(2) : "?") +
                  " · conf=" + (d.confidence != null ? d.confidence.toFixed(2) : "?");
      let candList = "";
      if (d.decision === "REVIEW" && d.candidates && d.candidates.length) {
        candList = '<div class="cmp-cand">үміткерлер: ' +
                   d.candidates.slice(0, 3).map((c) => esc(c.word) + " (" +
                   (c.prob != null ? (c.prob * 100).toFixed(0) : "?") + "%)").join(", ") +
                   "</div>";
      }
      html += '<span class="' + cls + '" title="' + esc(title) + '">' +
              esc(emitted) + "</span>" + candList;
      cursor = idx + d.token.length;
    });
    if (cursor < text.length) html += esc(text.slice(cursor));
    meta.innerHTML = '<div class="cmp-out">' + html + "</div>";
    const st = data.stats || {};
    metaBar.textContent = "KEEP: " + (st.keep || 0) +
                          " · CORRECT: " + (st.correct || 0) +
                          " · REVIEW: " + (st.review || 0);
    return { changed: st.correct || 0, review: st.review || 0 };
  }

  async function callEndpoint(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      let detail = resp.status + " " + resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch (_) {}
      return { error: detail };
    }
    return resp.json();
  }

  async function run() {
    const text = document.getElementById("cmpInput").value;
    const btn = document.getElementById("cmpRun");
    const stats = document.getElementById("cmpStats");
    if (!text.trim()) return;
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = "…орындалуда";
    stats.textContent = "";
    document.getElementById("cmpOld").innerHTML = '<span class="ph">…</span>';
    document.getElementById("cmpNew").innerHTML = '<span class="ph">…</span>';
    document.getElementById("cmpOldMeta").textContent = "";
    document.getElementById("cmpNewMeta").textContent = "";
    try {
      // fire both requests in parallel
      const [oldRes, newRes] = await Promise.all([
        callEndpoint("/api/v1/correct", { text: text }),
        callEndpoint("/api/v1/correct3", { text: text }),
      ]);
      const o = renderOld(oldRes, text);
      const n = renderNew(newRes, text);
      stats.textContent = "ескі " + o.changed + " өзгертті  ·  жаңа " +
                         n.changed + " түзетті" +
                         (n.review ? " · " + n.review + " REVIEW" : "");
    } catch (e) {
      document.getElementById("cmpOld").innerHTML =
        '<span class="ph">қате: ' + esc(e.message) + "</span>";
    }
    btn.disabled = false;
    btn.textContent = orig;
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("cmpRun").addEventListener("click", run);
    document.getElementById("cmpSample").addEventListener("click", function () {
      document.getElementById("cmpInput").value = SAMPLES[sampleIdx % SAMPLES.length];
      sampleIdx++;
      run();
    });
    document.getElementById("cmpClear").addEventListener("click", function () {
      document.getElementById("cmpInput").value = "";
      document.getElementById("cmpOld").innerHTML = '<span class="ph">нәтиже осында…</span>';
      document.getElementById("cmpNew").innerHTML = '<span class="ph">нәтиже осында…</span>';
      document.getElementById("cmpOldMeta").textContent = "";
      document.getElementById("cmpNewMeta").textContent = "";
      document.getElementById("cmpStats").textContent = "";
    });
    // Ctrl/Cmd+Enter runs the comparison
    document.getElementById("cmpInput").addEventListener("keydown", function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); }
    });
  });
})();
