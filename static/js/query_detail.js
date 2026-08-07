/**
 * KazMorphCorpus-2026 — Saved-query detail (route /dashboard/query/{id}).
 *
 * Renders a stored analysis from window.SAVED_RESULT/SAVED_TEXT injected
 * server-side. Self-contained: pure render helpers, no /api/analyze wiring,
 * no input/analytics state. Mirrors the rendering shape of dashboard.js so
 * the saved result looks identical to a live analysis.
 */

// ============================================================
//  State — populated from the server-injected globals.
// ============================================================
let lastResult = window.SAVED_RESULT || [];

document.addEventListener("DOMContentLoaded", () => {
    // Language switcher (rendered by the shared chrome macro).
    document.querySelectorAll(".lang-btn").forEach(btn => {
        btn.addEventListener("click", () => setLang(btn.dataset.lang));
    });
    applyLang();

    const text = window.SAVED_TEXT || "";

    renderHighlighted(text, lastResult);
    renderTable(lastResult);
    renderStats(lastResult);

    // Token detail: click a token in the highlighted text.
    document.getElementById("highlighted").addEventListener("click", e => {
        const tokEl = e.target.closest(".tok");
        if (tokEl && tokEl.dataset.idx !== undefined) showTokenDetail(parseInt(tokEl.dataset.idx));
    });
    // Token detail: click a table row.
    document.getElementById("tbody").addEventListener("click", e => {
        const row = e.target.closest("tr.row-clickable");
        if (row && row.dataset.idx !== undefined) showTokenDetail(parseInt(row.dataset.idx));
    });

    document.getElementById("sheetClose").addEventListener("click", closeSheet);
    document.getElementById("sheetOverlay").addEventListener("click", closeSheet);
});

// ============================================================
//  Render — classic stat cards (BOR/AFC/SEG/AGR counts).
//  Named renderStats so i18n.js#applyLang re-renders on language switch.
// ============================================================
function renderStats(tokens) {
    const counts = { BOR: 0, AFC: 0, SEG: 0, AGR: 0 };
    tokens.forEach(tok => {
        if (counts.hasOwnProperty(tok.difficulty_class)) counts[tok.difficulty_class]++;
    });
    document.getElementById("stat-bor").textContent = counts.BOR;
    document.getElementById("stat-afc").textContent = counts.AFC;
    document.getElementById("stat-seg").textContent = counts.SEG;
    document.getElementById("stat-agr").textContent = counts.AGR;
    document.getElementById("stats").style.display = "flex";
}

// ============================================================
//  Render — highlighted text
// ============================================================
function renderHighlighted(text, tokens) {
    const container = document.getElementById("highlighted");
    let html = "";
    let pos = 0;
    tokens.forEach((tok, i) => {
        const idx = text.indexOf(tok.token, pos);
        if (idx === -1) { html += escapeHtml(tok.token) + " "; return; }
        if (idx > pos) html += escapeHtml(text.substring(pos, idx));
        const cls = tok.difficulty_class.toLowerCase();
        const confPct = (tok.confidence * 100).toFixed(0);
        const hasSugg = tok.suggestions && tok.suggestions.length > 0;
        const suggTip = hasSugg ? `<br>→ ${tok.suggestions.join(", ")}` : "";
        html += `<span class="tok ${cls}${hasSugg ? " has-sugg" : ""}" data-idx="${i}">${escapeHtml(tok.token)}` +
            `<span class="tip">${tok.pos} | ${tok.lemma} | ${confPct}%<br>${escapeHtml(getDescription(tok))}${suggTip}</span>` +
            `</span>`;
        pos = idx + tok.token.length;
    });
    if (pos < text.length) html += escapeHtml(text.substring(pos));
    container.innerHTML = html;
}

// ============================================================
//  Render — token table
// ============================================================
function renderTable(tokens) {
    const tbody = document.getElementById("tbody");
    let html = "";
    tokens.forEach((tok, i) => {
        const confPct = (tok.confidence * 100).toFixed(1);
        const confHue = Math.round(tok.confidence * 120); // 0=red → 120=green
        const confColor = `hsl(${confHue}, 70%, 45%)`;
        const tagClass = "tag tag-" + tok.difficulty_class.toLowerCase();
        const suffixes = tok.suffixes.length > 0
            ? tok.suffixes.map(s =>
                `<span style="background:var(--primary-bg);color:var(--primary);padding:1px 7px;border-radius:9999px;font-size:12px">${escapeHtml(s)}</span>`
              ).join(" ")
            : "—";
        html += `<tr class="row-clickable" data-idx="${i}">
            <td>${i + 1}</td>
            <td><strong>${escapeHtml(tok.token)}</strong></td>
            <td>${escapeHtml(tok.lemma)}</td>
            <td><span style="background:var(--tag-none-bg);padding:1px 8px;border-radius:6px;font-size:12px">${translatePOS(tok.pos)}</span></td>
            <td><span class="conf-bar"><span class="conf-fill" style="width:${confPct}%;background:${confColor}"></span></span>${confPct}%</td>
            <td>${suffixes}</td>
            <td><span class="${tagClass}">${tok.difficulty_class}</span></td>
            <td style="font-size:12px">${tok.suggestions && tok.suggestions.length > 0
                ? tok.suggestions.map(s => `<span class="suggestion-chip">${escapeHtml(s)}</span>`).join(" ")
                : "—"}</td>
        </tr>`;
    });
    tbody.innerHTML = html;
    document.getElementById("tokenCount").textContent = tokens.length + " " + t("tokens");
}

// ============================================================
//  Token detail side sheet
// ============================================================
function showTokenDetail(idx) {
    if (!lastResult || !lastResult[idx]) return;
    const tok = lastResult[idx];
    const confPct = (tok.confidence * 100).toFixed(1);
    const confHue = Math.round(tok.confidence * 120);
    const confColor = `hsl(${confHue}, 70%, 45%)`;
    const tagClass = "tag tag-" + tok.difficulty_class.toLowerCase();

    const suffixHtml = tok.suffixes.length > 0
        ? tok.suffixes.map((s, i) => {
            const cat = tok.suffix_categories && tok.suffix_categories[i] ? translateCat(tok.suffix_categories[i]) : "";
            const label = cat ? `<span style="font-size:11px;color:var(--text-muted);margin-left:4px">${cat}</span>` : "";
            return `<span class="detail-suffix">${escapeHtml(s)}${label}</span>`;
          }).join("")
        : `<span style="color:var(--text-muted)">—</span>`;

    const suggHtml = tok.suggestions && tok.suggestions.length > 0
        ? tok.suggestions.map(s => `<span class="suggestion-chip">${escapeHtml(s)}</span>`).join(" ")
        : `<span style="color:var(--text-muted)">—</span>`;

    document.getElementById("sheetBody").innerHTML = `
        <div class="detail-card">
            <div class="detail-token">${escapeHtml(tok.token)}</div>
            <div class="detail-grid">
                <div class="detail-row"><span class="detail-label">${t("detail_lemma")}</span>
                    <span class="detail-value">${escapeHtml(tok.lemma)}</span></div>
                <div class="detail-row"><span class="detail-label">${t("detail_pos")}</span>
                    <span class="detail-value"><span class="detail-pos-tag">${translatePOS(tok.pos)} <span style="font-size:11px;color:var(--text-muted)">${tok.pos}</span></span></span></div>
                <div class="detail-row"><span class="detail-label">${t("detail_confidence")}</span>
                    <span class="detail-value"><span class="conf-bar" style="width:100px"><span class="conf-fill" style="width:${confPct}%;background:${confColor}"></span></span><strong>${confPct}%</strong></span></div>
                <div class="detail-row"><span class="detail-label">${t("detail_class")}</span>
                    <span class="detail-value"><span class="${tagClass}">${tok.difficulty_class}</span></span></div>
            </div>
            <div class="detail-section"><div class="detail-section-title">${t("detail_suffixes")}</div><div class="detail-suffixes">${suffixHtml}</div></div>
            <div class="detail-section"><div class="detail-section-title">${t("detail_description")}</div><p class="detail-desc">${escapeHtml(getDescription(tok))}</p></div>
            <div class="detail-section"><div class="detail-section-title">${t("detail_suggestions")}</div><div>${suggHtml}</div></div>
        </div>`;
    document.getElementById("sheetTitle").textContent = t("detail_sheet_title");
    document.getElementById("sheet").classList.add("open");
    document.getElementById("sheetOverlay").classList.add("open");
}

function closeSheet() {
    document.getElementById("sheet").classList.remove("open");
    document.getElementById("sheetOverlay").classList.remove("open");
}

// ============================================================
//  Localized helpers (mirror dashboard.js/app.js)
// ============================================================
function getDescription(tok) {
    const cls = tok.difficulty_class;
    const n = tok.suffixes ? tok.suffixes.length : 0;
    switch (cls) {
        case "BOR": return t("desc_bor");
        case "AFC": return t("desc_afc") + " (" + n + t("desc_afc_suffixes") + ")";
        case "SEG": return t("desc_seg");
        case "AGR": return t("desc_agr");
        default:    return t("desc_none");
    }
}
function translateCat(cat) { return t("cat_" + cat.replace(/\./g, "_")); }
function translatePOS(pos) { return t("pos_" + pos); }

// ============================================================
//  Utility
// ============================================================
function escapeHtml(str) {
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
