/**
 * KazMorphCorpus-2026 — Main application logic
 */

// ============================================================
//  State
// ============================================================
let lastResult = null;

// Sample texts (Kazakh)
const SAMPLES = [
    {
        label: "constitution",
        text: "Қазақстан Республикасының Конституциясы бойынша әрбір адамның " +
              "өмір сүруге, жеке басының бостандығына және қорғалуына құқығы бар. " +
              "Осы құқықтарды бұзуға ешкімнің құқығы жоқ."
    },
    {
        label: "education",
        text: "Білім беру жүйесі мемлекеттік және жекеменшік мекемелер арқылы " +
              "жүзеге асырылады. Оқушылардың білім алуға деген құқығы " +
              "Конституциямен кепілденген."
    },
    {
        label: "science",
        text: "Ғылым мен техниканың дамуы қоғамдық прогрессінің маңызды " +
              "қозғаушы күші болып табылады. Зерттеу жұмыстары " +
              "инновациялық технологияларды қалыптастырады."
    },
    {
        label: "culture",
        text: "Қазақстанның мәдени мұрасы ғасырлар бойы қалыптасқан " +
              "дәстүрлер мен әдет-ғұрыптарды қамтиды. Халықтың " +
              "рухани байлығы — оның тілі мен мәдениеті."
    },
    {
        label: "nature",
        text: "Қазақстанның табиғаты алуан түрлі: биік таулардан бастап " +
              "кең далаларға дейінгі ландшафттар кездеседі. " +
              "Жер бедерінің ерекшелігі климатқа тікелей әсер етеді."
    },
    {
        label: "history",
        text: "Ежелгі қазақ жерінде сақтар мен ғұндардың ұлы державалары " +
              "орналасқан болатын.Ұлы Жібек жолынің бір бөлігі осы " +
              "аймақ арқылы өткен."
    },
    {
        label: "economy",
        text: "Экономикалық дамудың негізгі бағыттары — индустриализация " +
              "және цифрландыру. Шетелдік инвестицияларды тарту " +
              "мемлекеттің басты міндеттерінің бірі."
    },
    {
        label: "sport",
        text: "Қазақстандық спортшылар халықаралық жарыстарда жоғары " +
              "жетістіктерге жетіп келеді. Бокс, дзюдо және " +
              "күрестен еліміздің спортшылары әлемге танымал."
    },
    {
        label: "literature",
        text: "Абай Құнанбаевтың шығармалары қазақ әдебиетінің алтын " +
              "қорынан орын алған. Оның өлеңдері мен қара сөздері " +
              "бүгінде бүкіл әлемге аударылған."
    },
    {
        label: "technology",
        text: "Астана халықаралық технологиялық парк ақпараттық " +
              "технологиялар саласындағы инновациялар орталығы " +
              "болып табылады. Жас кәсіпкерлер осында өз проектілерін іске асырады."
    },
];

// ============================================================
//  Init
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    // Language switcher buttons
    document.querySelectorAll(".lang-btn").forEach(btn => {
        btn.addEventListener("click", () => setLang(btn.dataset.lang));
    });

    // Apply saved language
    applyLang();

    // Load server health / stage status
    loadHealth();

    // Character counter
    document.getElementById("input").addEventListener("input", updateCharCount);
    updateCharCount();

    // Analyze button click
    document.getElementById("analyzeBtn").addEventListener("click", analyze);

    // Ctrl+Enter to analyze
    document.getElementById("input").addEventListener("keydown", e => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
    });

    // Side sheet: close button + overlay click
    document.getElementById("sheetClose").addEventListener("click", closeSheet);
    document.getElementById("sheetOverlay").addEventListener("click", closeSheet);

    // Token detail: click on .tok in highlighted text
    document.getElementById("highlighted").addEventListener("click", e => {
        const tokEl = e.target.closest(".tok");
        if (tokEl && tokEl.dataset.idx !== undefined) {
            showTokenDetail(parseInt(tokEl.dataset.idx));
        }
    });

    // Token detail: click on table row
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

        // Render active stages from server
        let html = data.stages.map(s =>
            `<span class="stage-badge">${s}</span>`
        ).join("");

        // Add inactive Neural stage
        html += '<span class="stage-badge inactive">Neural</span>';
        stagesEl.innerHTML = html;
    } catch {
        document.getElementById("stages").innerHTML =
            '<span class="stage-badge" style="background:#fce4ec;color:#c62828">Offline</span>';
    }
}

// ============================================================
//  Analyze — main action
// ============================================================
async function analyze() {
    const text = document.getElementById("input").value.trim();
    if (!text) return;

    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> ${t("btn_analyzing")}`;

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
        renderStats(data.tokens);
    } catch (e) {
        document.getElementById("highlighted").innerHTML =
            `<span class="placeholder" style="color:#dc3545">${t("error_prefix")} ${e.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.textContent = t("btn_analyze");
    }
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
        // Text between tokens (spaces, punctuation)
        if (idx > pos) {
            html += escapeHtml(text.substring(pos, idx));
        }
        const cls = tok.difficulty_class.toLowerCase();
        const confPct = (tok.confidence * 100).toFixed(0);
        const hasSugg = tok.suggestions && tok.suggestions.length > 0;
        const suggTip = hasSugg ? `<br>→ ${tok.suggestions.join(", ")}` : "";
        html += `<span class="tok ${cls}${hasSugg ? " has-sugg" : ""}" data-idx="${i}">${escapeHtml(tok.token)}` +
            `<span class="tip">${tok.pos} | ${tok.lemma} | ${confPct}%<br>${escapeHtml(getDescription(tok))}${suggTip}</span>` +
            `</span>`;
        pos = idx + tok.token.length;
    });

    if (pos < text.length) {
        html += escapeHtml(text.substring(pos));
    }

    container.innerHTML = html;
}

// ============================================================
//  Render — results table
// ============================================================
function renderTable(tokens) {
    const tbody = document.getElementById("tbody");
    let html = "";

    tokens.forEach((tok, i) => {
        const confPct = (tok.confidence * 100).toFixed(1);
        // Smooth color: 0% = red (hsl 0), 100% = green (hsl 120)
        const confHue = Math.round(tok.confidence * 120);
        const confColor = `hsl(${confHue}, 70%, 45%)`;
        const tagClass = "tag tag-" + tok.difficulty_class.toLowerCase();
        const suffixes = tok.suffixes.length > 0
            ? tok.suffixes.map(s =>
                `<span style="background:var(--primary-bg);color:var(--primary);padding:1px 6px;border-radius:3px;font-size:12px">` +
                `${escapeHtml(s)}</span>`
              ).join(" ")
            : "—";

        html += `<tr class="row-clickable" data-idx="${i}">
            <td>${i + 1}</td>
            <td><strong>${escapeHtml(tok.token)}</strong></td>
            <td>${escapeHtml(tok.lemma)}</td>
            <td><span style="background:var(--tag-none-bg);padding:1px 8px;border-radius:3px;font-size:12px">${translatePOS(tok.pos)}</span></td>
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
//  Render — stats cards
// ============================================================
function renderStats(tokens) {
    const counts = { BOR: 0, AFC: 0, SEG: 0, AGR: 0 };
    tokens.forEach(tok => {
        if (counts.hasOwnProperty(tok.difficulty_class)) {
            counts[tok.difficulty_class]++;
        }
    });

    document.getElementById("stat-bor").textContent = counts.BOR;
    document.getElementById("stat-afc").textContent = counts.AFC;
    document.getElementById("stat-seg").textContent = counts.SEG;
    document.getElementById("stat-agr").textContent = counts.AGR;
    document.getElementById("stats").style.display = "flex";
}

// ============================================================
//  Actions
// ============================================================
function clearAll() {
    document.getElementById("input").value = "";
    document.getElementById("highlighted").innerHTML =
        `<span class="placeholder" data-i18n="result_placeholder">${t("result_placeholder")}</span>`;
    document.getElementById("tbody").innerHTML = "";
    document.getElementById("stats").style.display = "none";
    document.getElementById("tokenCount").textContent = "";
    lastResult = null;
    updateCharCount();
}

function loadSample() {
    // Build sample cards
    const body = document.getElementById("sheetBody");
    body.innerHTML = SAMPLES.map((s, i) =>
        `<div class="sample-item" onclick="pickSample(${i})">` +
        `<span class="sample-label">${t("sample_" + s.label)}</span>` +
        `${escapeHtml(s.text)}` +
        `</div>`
    ).join("");

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
                ? translateCat(tok.suffix_categories[i])
                : "";
            const label = cat ? `<span style="font-size:11px;color:var(--text-muted);margin-left:4px">${cat}</span>` : "";
            return `<span class="detail-suffix">${escapeHtml(s)}${label}</span>`;
          }).join("")
        : `<span style="color:var(--text-muted)">—</span>`;

    const suggHtml = tok.suggestions && tok.suggestions.length > 0
        ? tok.suggestions.map(s => `<span class="suggestion-chip">${escapeHtml(s)}</span>`).join(" ")
        : `<span style="color:var(--text-muted)">—</span>`;

    const body = document.getElementById("sheetBody");
    body.innerHTML = `
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
        </div>
    `;

    document.getElementById("sheetTitle").textContent = t("detail_sheet_title");
    document.getElementById("sheet").classList.add("open");
    document.getElementById("sheetOverlay").classList.add("open");
}

// ============================================================
//  Localized description builder
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

/** Translate a grammatical category code: "POSS.3" -> cat_POSS_3 key -> "Притяж.3л." */
function translateCat(cat) {
    const key = "cat_" + cat.replace(/\./g, "_");
    return t(key);
}

/** Translate POS tag: NOUN -> pos_NOUN -> "Зат ес." */
function translatePOS(pos) {
    const key = "pos_" + pos;
    return t(key);
}

/** Build grammar HTML as pretty badges: "-тер-Множ.ч.  -іміз-Притяж.1л.мн.  -ден-Исх.п." */
function getGrammar(tok) {
    const sfx = tok.suffixes || [];
    const cat = tok.suffix_categories || [];
    if (sfx.length === 0) return "—";
    return sfx.map((s, i) => {
        const label = cat[i] ? translateCat(cat[i]) : "?";
        return '<span class="grammar-badge">-' + escapeHtml(s) + '- <span class="grammar-label">' + label + '</span></span>';
    }).join(" ");
}

// ============================================================
//  Utility
// ============================================================
function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
