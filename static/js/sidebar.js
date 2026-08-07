/**
 * KazMorphCorpus-2026 — shared shell logic (loaded on every page via base.html).
 *
 * Two responsibilities:
 *   1. fill #stages in the header from /api/health on pages whose server
 *      context omits `stages` (e.g. /dashboard, /login). The analyzer pages
 *      (/, /legacy) already call their own loadHealth(), so we skip if the
 *      badges are already populated.
 *   2. If a user is logged in, fetch /api/dashboard/stats and fill the
 *      left-sidebar stats block (#sbStats). Silent no-op on 401 / error.
 *
 * Does NOT depend on i18n.js being present (defensive). Uses bare strings
 * only for the offline badge.
 */
(function () {
    "use strict";

    // ---- 1. Pipeline stage badges --------------------------------------
    function fillStages() {
        const el = document.getElementById("stages");
        if (!el) return;
        // Skip if already rendered by the page's own script (/, /legacy).
        if (el.querySelector(".stage-badge")) return;

        fetch("/api/health")
            .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
            .then(function (data) {
                let html = (data.stages || []).map(function (s) {
                    return '<span class="stage-badge">' + s + "</span>";
                }).join("");
                if (!(data.stages || []).includes("Neural")) {
                    html += '<span class="stage-badge inactive">Neural</span>';
                }
                el.innerHTML = html || '<span class="stage-badge inactive">Offline</span>';
            })
            .catch(function () {
                el.innerHTML = '<span class="stage-badge inactive">Offline</span>';
            });
    }

    // ---- 2. Logged-in user stats ---------------------------------------
    function loadStats() {
        const box = document.getElementById("sbStats");
        if (!box) return;                 // not logged in (block is hidden)
        fetch("/api/dashboard/stats")
            .then(function (r) {
                if (r.status === 401) return Promise.reject(); // anonymous
                return r.ok ? r.json() : Promise.reject();
            })
            .then(function (data) {
                const q = document.getElementById("sbTotalQueries");
                const t = document.getElementById("sbTotalTokens");
                if (q) q.textContent = data.total_queries;
                if (t) t.textContent = data.total_tokens;
                box.hidden = false;
            })
            .catch(function () {
                /* leave the block hidden — graceful degradation */
            });
    }

    function init() {
        fillStages();
        loadStats();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
