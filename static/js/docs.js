/**
 * KazMorph — страница документации API (3-колоночный layout).
 *
 *  Левая колонка: липкая навигация (подсветка активного раздела).
 *  Центр:        описание эндпоинтов + таблицы параметров.
 *  Правая:       липкая панель кода — запрос/ответ, синхронизируется с
 *                активным эндпоинтом при скролле (IntersectionObserver).
 *
 *  Своя applyLang(), т.к. app.js здесь не загружается.
 */

/* Данные: что показывать в правой панели для каждого раздела.
   tabs — именованные вкладки; имя переводится через i18n-ключи tab_request/tab_response. */
const VIEWS = {
    overview: {
        path: null,
        tabs: { tab_request:
            'curl {{ base_url }}\n' +
            'curl -X POST {{ base_url }}/api/v1/correct \\\n' +
            '  -H "Content-Type: application/json" \\\n' +
            '  -d \'{"text": "Бүгін адамм келді"}\'' },
    },
    auth: {
        path: null,
        tabs: { tab_request:
            '# Если администратор задал KAZMORPH_API_KEY:\n' +
            'curl -H "X-API-Key: ВАШ_КЛЮЧ" {{ base_url }}/api/v1/info' },
    },
    quickstart: {
        path: "/api/v1/correct", method: "POST",
        tabs: {
            tab_request:
                'curl -X POST {{ base_url }}/api/v1/correct \\\n' +
                '  -H "Content-Type: application/json" \\\n' +
                '  -d \'{"text": "Бүгін адамм мектепке келді"}\'',
            tab_response:
                '{\n' +
                '  "corrected": "Бүгін адам мектепке келді",\n' +
                '  "corrections": [\n' +
                '    { "token": "адамм", "suggestion": "адам", "confidence": 0.40 }\n' +
                '  ],\n' +
                '  "stats": { "tokens": 4, "corrected": 1 }\n' +
                '}',
        },
    },
    correct: {
        path: "/api/v1/correct", method: "POST",
        tabs: {
            tab_request:
                '{\n' +
                '  "text": "Бүгін адамм мектепке келді",\n' +
                '  "confidence_threshold": 0.5\n' +
                '}',
            tab_response:
                '{\n' +
                '  "original": "Бүгін адамм мектепке келді",\n' +
                '  "corrected": "Бүгін адам мектепке келді",\n' +
                '  "corrections": [\n' +
                '    {\n' +
                '      "position": 6,\n' +
                '      "token": "адамм",\n' +
                '      "suggestion": "адам",\n' +
                '      "confidence": 0.40,\n' +
                '      "pos": "NOUN"\n' +
                '    }\n' +
                '  ],\n' +
                '  "stats": { "tokens": 4, "corrected": 1 }\n' +
                '}',
        },
    },
    batch: {
        path: "/api/v1/analyze/batch", method: "POST",
        tabs: {
            tab_request: '{ "texts": ["Қазақстан Республикасы", "адам сөз"] }',
            tab_response:
                '{\n' +
                '  "count": 2,\n' +
                '  "results": [\n' +
                '    {\n' +
                '      "text": "Қазақстан Республикасы",\n' +
                '      "tokens": [\n' +
                '        { "token": "Қазақстан", "lemma": "Қазақстан", "pos": "NOUN",\n' +
                '          "suffixes": [], "confidence": 0.65, "difficulty_class": "BOR",\n' +
                '          "suggestions": [] }\n' +
                '      ]\n' +
                '    },\n' +
                '    { "text": "адам сөз", "tokens": [ "..." ] }\n' +
                '  ]\n' +
                '}',
        },
    },
    analyze: {
        path: "/api/analyze", method: "POST",
        tabs: {
            tab_request: '{ "text": "Қазақстан Республикасының Конституциясы" }',
            tab_response:
                '{\n' +
                '  "tokens": [\n' +
                '    {\n' +
                '      "token": "Қазақстан",\n' +
                '      "lemma": "Қазақстан",\n' +
                '      "pos": "NOUN",\n' +
                '      "suffixes": [],\n' +
                '      "confidence": 0.65,\n' +
                '      "difficulty_class": "BOR",\n' +
                '      "description": "Borrowed word — morphological decomposition is harder",\n' +
                '      "suggestions": []\n' +
                '    }\n' +
                '  ]\n' +
                '}',
        },
    },
    evaluate: {
        path: "/api/v1/evaluate", method: "POST",
        tabs: {
            tab_request:
                '{\n' +
                '  "gold": "# sent_id = t:1\\n1\\tадам\\tадам\\tNOUN\\t...",\n' +
                '  "pred": "# sent_id = t:1\\n1\\tадам\\tадам\\tVERB\\t...",\n' +
                '  "min_support": 5\n' +
                '}',
            tab_response:
                '{\n' +
                '  "summary": {\n' +
                '    "gold_tokens": 3, "aligned_tokens": 3, "coverage": 1.0,\n' +
                '    "pos_accuracy": 0.6667, "macro_f1": 0.5556, "weighted_f1": 0.5556,\n' +
                '    "lemma_accuracy": 1.0, "pos_errors": 1\n' +
                '  },\n' +
                '  "per_tag": [\n' +
                '    { "tag": "NOUN", "precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1 }\n' +
                '  ],\n' +
                '  "top_confusions": [ { "gold": "NOUN", "pred": "VERB", "count": 1 } ],\n' +
                '  "recommendations": [ "..." ],\n' +
                '  "per_text": [ "..." ]\n' +
                '}',
        },
    },
    info: {
        path: "/api/v1/info", method: "GET",
        tabs: { tab_response:
            '{\n' +
            '  "name": "KazMorphCorpus-2026",\n' +
            '  "version": "1.0.0",\n' +
            '  "stages": ["RMA", "CRF", "Neural", "MFRN"],\n' +
            '  "models": { "crf": true, "neural": true },\n' +
            '  "lexicon": { "nouns": 13860, "verbs": 3319,\n' +
            '               "adjectives": 5207, "borrowings": 5481 }\n' +
            '}' },
    },
    health: {
        path: "/api/health", method: "GET",
        tabs: { tab_response:
            '{ "status": "ok", "stages": ["RMA", "CRF", "Neural", "MFRN"] }' },
    },
    integration: {
        path: null,
        tabs: {
            python:
                'import requests\n\n' +
                'BASE = "{{ base_url }}"\n' +
                'resp = requests.post(f"{BASE}/api/v1/correct",\n' +
                '                    json={"text": "Бүгін адамм келді"})\n' +
                'data = resp.json()\n' +
                'print(data["corrected"])    # Бүгін адам келді\n' +
                'print(data["corrections"])  # [{"token": "адамм", ...}]',
            javascript:
                'const BASE = "{{ base_url }}";\n' +
                'const resp = await fetch(`${BASE}/api/v1/correct`, {\n' +
                '  method: "POST",\n' +
                '  headers: { "Content-Type": "application/json" },\n' +
                '  body: JSON.stringify({ text: "Бүгін адамм келді" }),\n' +
                '});\n' +
                'const data = await resp.json();\n' +
                'console.log(data.corrected);   // Бүгін адам келді',
            curl:
                'curl -X POST {{ base_url }}/api/v1/correct \\\n' +
                '  -H "Content-Type: application/json" \\\n' +
                '  -d \'{"text": "Бүгін адамм келді"}\'',
        },
    },
    errors: {
        path: null,
        tabs: { tab_response:
            '// 422 — ошибка валидации (например, пустой texts)\n' +
            '// 401 — неверный или отсутствующий API-ключ\n' +
            '// 400 — не разобрано ни одного предложения (gold/pred)\n' +
            '// 503 — пайплайн не загружен' },
    },
};

let BASE_URL = location.origin;

/* ── Подсветка синтаксиса (JSON + код) ────────────────────────── */
function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightCode(code) {
    var esc = escapeHtml(code);
    var re = /(#[^\n]*|\/\/[^\n]*)|("(?:\\.|[^"\\])*"\s*:)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b-?\d+(?:\.\d+)?\b)|(\b(?:true|false|null|None|True|False)\b)|(\b(?:import|from|const|let|var|function|await|async|fetch|print|return|new|def|if|else|for|in|console|requests|await)\b)/g;
    return esc.replace(re, function (m, comment, key, str, num, bool, kw) {
        if (comment) return '<span class="hl-comment">' + comment + "</span>";
        if (key) {
            return key.replace(/("(?:\\.|[^"\\])*")(\s*:)/, '<span class="hl-key">$1</span>$2');
        }
        if (str) return '<span class="hl-string">' + str + "</span>";
        if (num) return '<span class="hl-number">' + num + "</span>";
        if (bool) return '<span class="hl-boolean">' + bool + "</span>";
        if (kw) return '<span class="hl-keyword">' + kw + "</span>";
        return m;
    });
}

function applyLang() {
    document.documentElement.lang = currentLang;
    document.title = t("docs_title");
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
        el.textContent = t(el.getAttribute("data-i18n"));
    });
    document.querySelectorAll(".lang-btn").forEach(function (btn) {
        btn.classList.toggle("active", btn.dataset.lang === currentLang);
    });
    // перерисовать панель кода (переводы названий вкладок)
    if (currentView) renderCodePanel(currentView);
}

let currentView = "overview";
let currentTab = null;   // активная вкладка в правой панели

function resolveBase(text) {
    return text.replace(/\{\{\s*base_url\s*\}\}/g, BASE_URL);
}

function renderCodePanel(viewKey) {
    currentView = viewKey;
    const view = VIEWS[viewKey];
    if (!view) return;

    const panel = document.getElementById("codePanel");
    const methodEl = document.getElementById("cpMethod");
    const pathEl = document.getElementById("cpPath");
    const tabsEl = document.getElementById("cpTabs");
    const codeEl = document.getElementById("cpCode");

    // шапка: метод + путь
    methodEl.textContent = view.method || "";
    methodEl.className = "method " + ((view.method || "").toLowerCase());
    methodEl.style.display = view.method ? "" : "none";
    pathEl.textContent = view.path || "";

    // вкладки
    const tabKeys = Object.keys(view.tabs);
    if (!currentTab || !view.tabs[currentTab]) currentTab = tabKeys[0];

    tabsEl.innerHTML = "";
    tabKeys.forEach(function (key) {
        const btn = document.createElement("button");
        btn.className = "code-tab" + (key === currentTab ? " active" : "");
        btn.textContent = t(key);
        btn.addEventListener("click", function () {
            currentTab = key;
            renderCodePanel(viewKey);
        });
        tabsEl.appendChild(btn);
    });
    tabsEl.style.display = tabKeys.length > 1 ? "" : "none";

    codeEl.innerHTML = highlightCode(resolveBase(view.tabs[currentTab]));
}

/* Копирование активной вкладки (кнопка в шапке панели) */
function initCopy() {
    document.getElementById("cpCopy").addEventListener("click", function () {
        const code = document.getElementById("cpCode").textContent;
        navigator.clipboard.writeText(code).then(function () {
            const btn = document.getElementById("cpCopy");
            btn.classList.add("copied");
            btn.textContent = t("docs_copied");
            setTimeout(function () {
                btn.classList.remove("copied");
                btn.textContent = t("docs_copy");
            }, 1200);
        });
    });
}

/* Подсветка навигации + синхронизация панели при скролле */
function initScrollSpy() {
    const sections = Array.from(document.querySelectorAll(".docs-main .ep, .docs-main .docs-h2"));
    // Собираем целевые: секции с data-view и заголовки-якоря (integration)
    const targets = Array.from(document.querySelectorAll("[data-view]"))
        .concat(Array.from(document.querySelectorAll(".docs-h2[id]")));

    const navLinks = {};
    document.querySelectorAll(".docs-nav .nav-item").forEach(function (a) {
        const id = a.getAttribute("href").replace("#", "");
        navLinks[id] = a;
    });

    const setActive = function (id) {
        Object.values(navLinks).forEach(function (a) { a.classList.remove("active"); });
        if (navLinks[id]) navLinks[id].classList.add("active");
        if (VIEWS[id]) {
            currentTab = null; // сброс вкладки при смене эндпоинта
            renderCodePanel(id);
        }
    };

    const io = new IntersectionObserver(function (entries) {
        // выбираем верхнюю видимую секцию
        const visible = entries
            .filter(function (e) { return e.isIntersecting; })
            .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
        if (visible.length) {
            const id = visible[0].target.id;
            if (id) setActive(id);
        }
    }, { rootMargin: "-20% 0px -70% 0px", threshold: 0 });

    targets.forEach(function (s) { if (s.id) io.observe(s); });
}

/* Плавный скролл по клику в навигации */
function initNavClicks() {
    document.querySelectorAll(".docs-nav a").forEach(function (a) {
        a.addEventListener("click", function (e) {
            const href = a.getAttribute("href");
            if (href.startsWith("#") && href.length > 1) {
                const el = document.querySelector(href);
                if (el) {
                    e.preventDefault();
                    el.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            }
        });
    });
}

/* Playground: /api/v1/correct */
function initPlayground() {
    const runBtn = document.getElementById("pgRun");
    const input = document.getElementById("pgInput");
    const output = document.getElementById("pgOutput");
    if (!runBtn) return;
    runBtn.addEventListener("click", async function () {
        const text = input.value;
        if (!text.trim()) return;
        runBtn.disabled = true;
        const orig = runBtn.textContent;
        runBtn.textContent = t("docs_pg_running");
        output.innerHTML = '<span class="placeholder">...</span>';
        try {
            const resp = await fetch("/api/v1/correct", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text }),
            });
            const data = await resp.json();
            output.innerHTML = highlightCode(JSON.stringify(data, null, 2));
        } catch (e) {
            output.innerHTML = '<span style="color:var(--tag-bor)">' +
                t("docs_pg_error") + ": " + e.message + "</span>";
        }
        runBtn.disabled = false;
        runBtn.textContent = orig;
    });
}

document.addEventListener("DOMContentLoaded", function () {
    BASE_URL = location.origin;
    applyLang();
    renderCodePanel("overview");
    initCopy();
    initScrollSpy();
    initNavClicks();
    initPlayground();
});
