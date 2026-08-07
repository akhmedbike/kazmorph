/**
 * KazMorphCorpus-2026 — Dashboard logic (route /).
 *
 * Self-contained: reuses the /api/analyze contract (9 token keys) and
 * /api/health, but does not depend on app.js (the legacy page script).
 *
 * NOTE on i18n coupling: i18n.js#applyLang() calls renderStats(lastResult),
 * renderTable(lastResult) and updateCharCount() as globals. We define all
 * three here so language switching re-renders the dashboard correctly.
 */

// ============================================================
//  State
// ============================================================
let lastResult = null;

// Sample texts (same set as the legacy page).
const SAMPLES = [
    { label: "constitution", text: "Қазақстан Республикасының Конституциясы бойынша әрбір адамның " +
          "өмір сүруге, жеке басының бостандығына және қорғалуына құқығы бар. " +
          "Осы құқықтарды бұзуға ешкімнің құқығы жоқ." },
    { label: "education", text: "Білім беру жүйесі мемлекеттік және жекеменшік мекемелер арқылы " +
          "жүзеге асырылады. Оқушылардың білім алуға деген құқығы " +
          "Конституциямен кепілденген." },
    { label: "science", text: "Ғылым мен техниканың дамуы қоғамдық прогрессінің маңызды " +
          "қозғаушы күші болып табылады. Зерттеу жұмыстары " +
          "инновациялық технологияларды қалыптастырады." },
    { label: "culture", text: "Қазақстанның мәдени мұрасы ғасырлар бойы қалыптасқан " +
          "дәстүрлер мен әдет-ғұрыптарды қамтиды. Халықтың " +
          "рухани байлығы — оның тілі мен мәдениеті." },
    { label: "nature", text: "Қазақстанның табиғаты алуан түрлі: биік таулардан бастап " +
          "кең далаларға дейінгі ландшафттар кездеседі. " +
          "Жер бедерінің ерекшелігі климатқа тікелей әсер етеді." },
    { label: "history", text: "Ежелгі қазақ жерінде сақтар мен ғұндардың ұлы державалары " +
          "орналасқан болатын.Ұлы Жібек жолынің бір бөлігі осы " +
          "аймақ арқылы өткен." },
    { label: "economy", text: "Экономикалық дамудың негізгі бағыттары — индустриализация " +
          "және цифрландыру. Шетелдік инвестицияларды тарту " +
          "мемлекеттің басты міндеттерінің бірі." },
    { label: "sport", text: "Қазақстандық спортшылар халықаралық жарыстарда жоғары " +
          "жетістіктерге жетіп келеді. Бокс, дзюдо және " +
          "күрестен еліміздің спортшылары әлемге танымал." },
    { label: "literature", text: "Абай Құнанбаевтың шығармалары қазақ әдебиетінің алтын " +
          "қорынан орын алған. Оның өлеңдері мен қара сөздері " +
          "бүгінде бүкіл әлемге аударылған." },
    { label: "technology", text: "Астана халықаралық технологиялық парк ақпараттық " +
          "технологиялар саласындағы инновациялар орталығы " +
          "болып табылады. Жас кәсіпкерлер осында өз проектілерін іске асырады." },
];

// Difficulty classes in display order. NONE = standard analysis.
const CLASSES = ["BOR", "AFC", "SEG", "AGR", "NONE"];

// ============================================================
//  Init
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    // Language switcher (rendered by chrome macro).
    document.querySelectorAll(".lang-btn").forEach(btn => {
        btn.addEventListener("click", () => setLang(btn.dataset.lang));
    });

    applyLang();
    loadHealth();

    const input = document.getElementById("input");
    input.addEventListener("input", updateCharCount);
    input.addEventListener("keydown", e => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
    });
    updateCharCount();

    document.getElementById("analyzeBtn").addEventListener("click", analyze);
    document.getElementById("clearBtn").addEventListener("click", clearAll);
    document.getElementById("sampleBtn").addEventListener("click", loadSample);

    document.getElementById("sheetClose").addEventListener("click", closeSheet);
    document.getElementById("sheetOverlay").addEventListener("click", closeSheet);

    // Token detail: click a token in the highlighted text.
    document.getElementById("highlighted").addEventListener("click", e => {
        const tokEl = e.target.closest(".tok");
        if (tokEl && tokEl.dataset.idx !== undefined) {
            showTokenDetail(parseInt(tokEl.dataset.idx));
        }
    });
    // Token detail: click a table row.
    document.getElementById("tbody").addEventListener("click", e => {
        const row = e.target.closest("tr.row-clickable");
        if (row && row.dataset.idx !== undefined) {
            showTokenDetail(parseInt(row.dataset.idx));
        }
    });
});

// ============================================================
//  UI helpers
// ============================================================
function updateCharCount() {
    const text = document.getElementById("input").value;
    document.getElementById("charCount").textContent = text.length + " " + t("chars");
}

async function loadHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const stagesEl = document.getElementById("stages");
        let html = data.stages.map(s => `<span class="stage-badge">${s}</span>`).join("");
        // Mark Neural inactive if the backend didn't report it as active.
        if (!data.stages.includes("Neural")) {
            html += '<span class="stage-badge inactive">Neural</span>';
        }
        stagesEl.innerHTML = html;
    } catch {
        document.getElementById("stages").innerHTML =
            '<span class="stage-badge inactive">Offline</span>';
    }
}

// ============================================================
//  Analyze
// ============================================================
async function analyze() {
    const text = document.getElementById("input").value.trim();
    if (!text) return;

    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${t("btn_analyzing")}`;

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        lastResult = data.tokens;

        renderHighlighted(text, data.tokens);
        renderTable(data.tokens);
        renderStats(data.tokens);   // KPIs + distributions
        // Reveal the below-the-fold results (KPI / distributions / table)
        // that were hidden until the first analysis.
        document.getElementById("dashResults").hidden = false;
    } catch (e) {
        document.getElementById("highlighted").innerHTML =
            `<span class="placeholder" style="color:var(--error)">${t("error_prefix")} ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("btn_analyze");
    }
}

// ============================================================
//  Render — KPIs + distributions
//  (Named renderStats so i18n.js#applyLang re-renders on language switch.)
// ============================================================
function renderStats(tokens) {
    const total = tokens.length;

    // Average confidence excluding PUNCT (punctuation is always 1.0 and skews
    // the mean). Falls back to "—" if there are no scorable tokens.
    const scorable = tokens.filter(tok => tok.pos !== "PUNCT");
    const avg = scorable.length > 0
        ? Math.round((scorable.reduce((s, tok) => s + tok.confidence, 0) / scorable.length) * 100)
        : null;

    // Share of genuinely uncertain tokens (confidence < 0.70), the same
    // threshold the active-learning loop uses to flag candidates for human
    // review (ACTIVE_LEARNING_THRESHOLD in kazmorph/pipeline.py). Computed on
    // the same scorable sample as `avg` so the two metrics stay consistent.
    // Shows as a percentage; "—" when there is nothing to score.
    const lowConf = scorable.length > 0
        ? Math.round((scorable.filter(tok => tok.confidence < 0.70).length / scorable.length) * 100)
        : null;

    const counts = { BOR: 0, AFC: 0, SEG: 0, AGR: 0, NONE: 0 };
    tokens.forEach(tok => {
        if (counts.hasOwnProperty(tok.difficulty_class)) counts[tok.difficulty_class]++;
    });
    const complex = counts.BOR + counts.AFC + counts.SEG + counts.AGR;

    document.getElementById("kpiTokens").textContent = total;
    document.getElementById("kpiConf").textContent = avg === null ? "—" : avg + "%";
    document.getElementById("kpiComplex").textContent = complex;
    document.getElementById("kpiBor").textContent = counts.BOR;
    document.getElementById("kpiLowConf").textContent = lowConf === null ? "—" : lowConf + "%";

    renderDifficulty(counts, total);
    renderPOS(tokens);
}

function renderDifficulty(counts, total) {
    const el = document.getElementById("distDifficulty");
    if (total === 0) {
        el.innerHTML = `<div class="dist-empty">${t("dashboard_empty")}</div>`;
        return;
    }
    // Skip classes with zero count for a cleaner chart.
    el.innerHTML = CLASSES
        .filter(c => counts[c] > 0)
        .map(c => {
            const pct = total > 0 ? (counts[c] / total) * 100 : 0;
            const cls = c.toLowerCase();
            const label = c === "NONE" ? t("stat_none") : t("stat_" + cls);
            return `<div class="dist-row">
                <span class="dist-name">${label}</span>
                <span class="dist-track"><span class="dist-fill ${cls}" style="width:${pct}%"></span></span>
                <span class="dist-count">${counts[c]}</span>
            </div>`;
        }).join("");
}

function renderPOS(tokens) {
    const el = document.getElementById("distPOS");
    if (tokens.length === 0) {
        el.innerHTML = `<div class="dist-empty">${t("dashboard_empty")}</div>`;
        return;
    }
    // Tally POS tags, then render descending by count.
    const counts = {};
    tokens.forEach(tok => { counts[tok.pos] = (counts[tok.pos] || 0) + 1; });
    const total = tokens.length;
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);

    el.innerHTML = entries.map(([pos, count]) => {
        const pct = (count / total) * 100;
        const label = (typeof t("pos_" + pos) === "string" && t("pos_" + pos) !== "pos_" + pos)
            ? t("pos_" + pos)
            : pos;
        return `<div class="dist-row">
            <span class="dist-name">${label}</span>
            <span class="dist-track"><span class="dist-fill pos" style="width:${pct}%"></span></span>
            <span class="dist-count">${count}</span>
        </div>`;
    }).join("");
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
        if (idx === -1) {
            html += escapeHtml(tok.token) + " ";
            return;
        }
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
//  Render — table
// ============================================================
function renderTable(tokens) {
    const tbody = document.getElementById("tbody");
    let html = "";

    tokens.forEach((tok, i) => {
        const confPct = (tok.confidence * 100).toFixed(1);
        const confHue = Math.round(tok.confidence * 120);     // 0=red → 120=green
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
            <td>
                <span class="conf-bar"><span class="conf-fill" style="width:${confPct}%;background:${confColor}"></span></span>
                ${confPct}%
            </td>
            <td>${suffixes}</td>
            <td><span class="${tagClass}">${tok.difficulty_class}</span></td>
            <td style="font-size:12px">${tok.suggestions && tok.suggestions.length > 0
                ? tok.suggestions.map(s => `<span class="suggestion-chip">${escapeHtml(s)}</span>`).join(" ")
                : "—"
            }</td>
        </tr>`;
    });

    tbody.innerHTML = html;
    document.getElementById("tokenCount").textContent = tokens.length + " " + t("tokens");
}

// ============================================================
//  Actions
// ============================================================
function clearAll() {
    document.getElementById("input").value = "";
    document.getElementById("highlighted").innerHTML =
        `<span class="placeholder">${t("result_placeholder")}</span>`;
    document.getElementById("tbody").innerHTML = "";
    document.getElementById("tokenCount").textContent = "";
    // Reset KPI cards and distributions to empty state.
    document.getElementById("kpiTokens").textContent = "—";
    document.getElementById("kpiConf").textContent = "—";
    document.getElementById("kpiComplex").textContent = "—";
    document.getElementById("kpiBor").textContent = "—";
    document.getElementById("kpiLowConf").textContent = "—";
    document.getElementById("distDifficulty").innerHTML = `<div class="dist-empty">${t("dashboard_empty")}</div>`;
    document.getElementById("distPOS").innerHTML = `<div class="dist-empty">${t("dashboard_empty")}</div>`;
    // Re-hide the below-the-fold results until the next analysis.
    document.getElementById("dashResults").hidden = true;
    lastResult = null;
    updateCharCount();
}

function loadSample() {
    const body = document.getElementById("sheetBody");
    body.innerHTML = SAMPLES.map((s, i) =>
        `<div class="sample-item" onclick="pickSample(${i})">` +
        `<span class="sample-label">${t("sample_" + s.label)}</span>` +
        `${escapeHtml(s.text)}` +
        `</div>`
    ).join("");
    document.getElementById("sheetTitle").textContent = t("sheet_title");
    document.getElementById("sheet").classList.add("open");
    document.getElementById("sheetOverlay").classList.add("open");
}

function closeSheet() {
    document.getElementById("sheet").classList.remove("open");
    document.getElementById("sheetOverlay").classList.remove("open");
}

function pickSample(idx) {
    document.getElementById("input").value = SAMPLES[idx].text;
    updateCharCount();
    closeSheet();
}

// ============================================================
//  Token detail sheet
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
            const cat = tok.suffix_categories && tok.suffix_categories[i]
                ? translateCat(tok.suffix_categories[i]) : "";
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
                <div class="detail-row">
                    <span class="detail-label">${t("detail_lemma")}</span>
                    <span class="detail-value">${escapeHtml(tok.lemma)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">${t("detail_pos")}</span>
                    <span class="detail-value"><span class="detail-pos-tag">${translatePOS(tok.pos)} <span style="font-size:11px;color:var(--text-muted)">${tok.pos}</span></span></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">${t("detail_confidence")}</span>
                    <span class="detail-value">
                        <span class="conf-bar" style="width:100px"><span class="conf-fill" style="width:${confPct}%;background:${confColor}"></span></span>
                        <strong>${confPct}%</strong>
                    </span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">${t("detail_class")}</span>
                    <span class="detail-value"><span class="${tagClass}">${tok.difficulty_class}</span></span>
                </div>
            </div>
            <div class="detail-section">
                <div class="detail-section-title">${t("detail_suffixes")}</div>
                <div class="detail-suffixes">${suffixHtml}</div>
            </div>
            <div class="detail-section">
                <div class="detail-section-title">${t("detail_description")}</div>
                <p class="detail-desc">${escapeHtml(getDescription(tok))}</p>
            </div>
            <div class="detail-section">
                <div class="detail-section-title">${t("detail_suggestions")}</div>
                <div>${suggHtml}</div>
            </div>
        </div>`;

    document.getElementById("sheetTitle").textContent = t("detail_sheet_title");
    document.getElementById("sheet").classList.add("open");
    document.getElementById("sheetOverlay").classList.add("open");
}

// ============================================================
//  Localized helpers
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

function translateCat(cat) {
    return t("cat_" + cat.replace(/\./g, "_"));
}

function translatePOS(pos) {
    return t("pos_" + pos);
}

// ============================================================
//  Utility
// ============================================================
function escapeHtml(str) {
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
