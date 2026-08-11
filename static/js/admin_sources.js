/* Admin sources / Active Learning: запуск шагов цикла + polling статуса.

  Шаги: harvest → select → review.
  POST /admin/sources/run {step} → 202 + state.
  GET  /admin/sources/status → snapshot (poll каждые 2 сек, пока running).
  GET  /admin/sources/candidates/preview → {n_candidates, est_cost_usd}
    (для confirm-диалога перед шагом review). */
(function () {
    "use strict";

    const POLL_MS = 2000;
    let pollTimer = null;

    const STEP_LABELS = {
        harvest: "сбор текстов",
        select: "отбор кандидатов",
        review: "LLM-ревью",
    };

    const adminSources = {
        /**
         * Запустить шаг цикла.
         * Для review — сначала превью кандидатов + confirm с оценкой стоимости.
         */
        run(step) {
            if (step === "review") {
                // Превью кандидатов может быть устаревшим (собрали после отрисовки) —
                // перечитаем с сервера, чтобы оценка стоимости была точной.
                return adminSources.previewAndConfirm();
            }
            return adminSources.launch(step);
        },

        /**
         * Шаг review: получить превью, показать confirm, при ОК — запустить.
         */
        previewAndConfirm() {
            fetch("/admin/sources/candidates/preview", { headers: { "Accept": "application/json" } })
                .then((r) => r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)))
                .then((pv) => {
                    adminSources.updateCandidatesPreview(pv);
                    if (!pv.n_candidates || pv.n_candidates === 0) {
                        alert("Нет кандидатов для LLM-ревью. Сначала запустите шаг 2 (отбор).");
                        return;
                    }
                    const cost = (pv.est_cost_usd || 0).toFixed(4);
                    const ok = confirm(
                        "LLM-ревью обработает " + pv.n_candidates + " кандидатов за " + pv.date +
                        ".\nОценка стоимости: ~$" + cost + " (Claude).\n\nЗапустить?"
                    );
                    if (ok) adminSources.launch("review");
                })
                .catch((err) => alert("Не удалось получить превью кандидатов: " + err.message));
        },

        /**
         * Фактический POST на запуск шага. Блокирует все кнопки на время выполнения.
         */
        launch(step) {
            const body = { step: step };
            if (step === "harvest") {
                const checked = document.querySelector(".source-radio:checked");
                body.source = checked && checked.value ? checked.value : null;
            }
            adminSources.setButtonsDisabled(true);

            fetch("/admin/sources/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            })
                .then((r) => {
                    if (r.status === 409) {
                        throw new Error("Шаг уже выполняется — дождитесь завершения");
                    }
                    if (r.status === 422) {
                        return r.json().then((d) => { throw new Error(d.detail || "Неверный запрос"); });
                    }
                    if (!r.ok) throw new Error("HTTP " + r.status);
                    return r.json();
                })
                .then((state) => {
                    adminSources.render(state);
                    adminSources.startPolling();
                })
                .catch((err) => {
                    alert("Не удалось запустить: " + err.message);
                    adminSources.setButtonsDisabled(false);
                });
        },

        // ── Polling ────────────────────────────────────────────────

        startPolling() {
            if (pollTimer) return;
            pollTimer = setInterval(adminSources.poll, POLL_MS);
            adminSources.poll();
        },

        stopPolling() {
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
        },

        poll() {
            fetch("/admin/sources/status", { headers: { "Accept": "application/json" } })
                .then((r) => r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)))
                .then((state) => {
                    adminSources.render(state);
                    if (state.status !== "running") {
                        adminSources.stopPolling();
                        adminSources.setButtonsDisabled(false);
                        // После done перечитаем превью кандидатов — оно могло измениться.
                        if (state.status === "done") adminSources.refreshPreview();
                    }
                })
                .catch((err) => console.warn("AL status poll failed:", err));
        },

        // ── Рендер ─────────────────────────────────────────────────

        /**
         * Перерисовать блок #al-status + индикатор шага в pipeline-steps.
         */
        render(state) {
            const box = document.getElementById("al-status");
            if (box) {
                box.setAttribute("data-state", state.status);
                box.setAttribute("data-step", state.step || "");
            }
            adminSources.highlightStep(state.status, state.step);

            if (state.status === "running") {
                const label = STEP_LABELS[state.step] || state.step;
                if (box) box.innerHTML = '<div class="al-running"><span class="spinner"></span>Выполняется: ' +
                    adminSources.esc(label) + "…</div>";
                return;
            }
            if (state.status === "done") {
                if (box) box.innerHTML = adminSources.renderSummary(state.step, state.result);
                return;
            }
            if (state.status === "error") {
                if (box) box.innerHTML = '<div class="al-error">Ошибка: ' +
                    adminSources.esc(state.result || "неизвестная") + "</div>";
                return;
            }
            // idle — пусто.
            if (box) box.innerHTML = "";
        },

        /**
         * Подсветить активный шаг в полосе ① → ② → ③.
         */
        highlightStep(status, step) {
            document.querySelectorAll(".pipeline-steps .pstep[data-step]").forEach(function (el) {
                el.classList.remove("active", "done");
                if (status === "running" && el.getAttribute("data-step") === step) {
                    el.classList.add("active");
                }
                if (status === "done" && el.getAttribute("data-step") === step) {
                    el.classList.add("done");
                }
            });
        },

        /**
         * Сводка завершённого шага — разная для harvest/select/review.
         */
        renderSummary(step, result) {
            if (step === "harvest") return adminSources.renderHarvest(result);
            if (step === "select") return adminSources.renderSelect(result);
            if (step === "review") return adminSources.renderReview(result);
            return '<div class="al-done">Готово.</div>';
        },

        renderHarvest(result) {
            if (!result || !result.sources) return '<div class="al-done">Сбор завершён.</div>';
            const rows = result.sources.map(function (s) {
                if (s.error) {
                    return '<tr class="sum-row-error"><td>' + adminSources.esc(s.name) +
                        '</td><td colspan="2">ошибка: ' + adminSources.esc(s.error) + "</td></tr>";
                }
                return '<tr><td>' + adminSources.esc(s.name) +
                    '</td><td class="num">' + (s.new || 0) +
                    '</td><td class="num">' + (s.dup || 0) + "</td></tr>";
            }).join("");
            return '<div class="al-done"><h4>Сбор завершён</h4>' +
                '<table class="al-summary"><thead><tr>' +
                "<th>Источник</th><th>Новых</th><th>Дубликатов</th>" +
                "</tr></thead><tbody>" + rows + "</tbody>" +
                '<tfoot><tr><td>Итого</td>' +
                '<td class="num">' + (result.total_new || 0) + "</td>" +
                '<td class="num">' + (result.total_skipped_dup || 0) + "</td>" +
                "</tr></tfoot></table>" +
                '<div class="al-next-hint">Далее: шаг 2 — отбор кандидатов.</div></div>';
        },

        renderSelect(result) {
            if (!result) return '<div class="al-done">Отбор завершён.</div>';
            return '<div class="al-done"><h4>Отбор завершён</h4>' +
                '<table class="al-summary al-summary-simple"><tbody>' +
                '<tr><th>Дата</th><td>' + adminSources.esc(result.date) + "</td></tr>" +
                '<tr><th>Статей обработано</th><td class="num">' + (result.n_articles || 0) + "</td></tr>" +
                '<tr><th>Токенов</th><td class="num">' + (result.n_tokens || 0) + "</td></tr>" +
                '<tr><th>Кандидатов отобрано</th><td class="num strong">' + (result.n_candidates || 0) + "</td></tr>" +
                "</tbody></table>" +
                '<div class="al-next-hint">Далее: шаг 3 — LLM-ревью этих кандидатов.</div></div>';
        },

        renderReview(result) {
            if (!result) return '<div class="al-done">LLM-ревью завершён.</div>';
            const verdicts = result.verdicts || {};
            const vrows = Object.keys(verdicts).map(function (k) {
                return '<tr><td>' + adminSources.esc(k) + '</td><td class="num">' + verdicts[k] + "</td></tr>";
            }).join("");
            return '<div class="al-done"><h4>LLM-ревью завершено</h4>' +
                '<table class="al-summary al-summary-simple"><tbody>' +
                '<tr><th>Кандидатов обработано</th><td class="num strong">' + (result.n_reviewed || 0) + "</td></tr>" +
                '<tr><th>Стоимость</th><td class="num">$' + (result.total_cost_usd || 0).toFixed(4) + "</td></tr>" +
                '<tr><th>Provider / модель</th><td>' + adminSources.esc(result.provider || "") + " / " +
                    adminSources.esc(result.model || "") + "</td></tr>" +
                "</tbody></table>" +
                (vrows ? '<table class="al-summary al-summary-simple"><thead><tr><th>Verdict</th><th class="num">Кол-во</th></tr></thead>' +
                    '<tbody>' + vrows + "</tbody></table>" : "") +
                '<div class="al-next-hint"><a href="/admin/review?status=pending">→ Перейти к ревью (pending)</a></div></div>';
        },

        // ── Превью кандидатов ──────────────────────────────────────

        refreshPreview() {
            fetch("/admin/sources/candidates/preview", { headers: { "Accept": "application/json" } })
                .then((r) => r.ok ? r.json() : null)
                .then((pv) => { if (pv) adminSources.updateCandidatesPreview(pv); })
                .catch(() => {});
        },

        updateCandidatesPreview(pv) {
            const box = document.getElementById("candidates-preview");
            if (!box) return;
            box.setAttribute("data-n", pv.n_candidates);
            box.setAttribute("data-cost", pv.est_cost_usd);
            box.innerHTML = 'Кандидатов за сегодня: <strong>' + pv.n_candidates + "</strong>" +
                ' <span class="cost-estimate">оценка: ~$' + (pv.est_cost_usd || 0).toFixed(4) + "</span>";
            // Кнопка review активна только если есть кандидаты.
            const btn = document.querySelector('.al-run-btn[data-step="review"]');
            if (btn) {
                if (pv.n_candidates > 0) {
                    btn.disabled = false;
                    btn.removeAttribute("title");
                } else {
                    btn.disabled = true;
                    btn.setAttribute("title", "Сначала отберите кандидатов (шаг 2)");
                }
            }
            // Кнопка «Показать кандидатов» — туда же.
            const showBtn = document.querySelector(".al-show-candidates");
            if (showBtn) {
                showBtn.disabled = pv.n_candidates === 0;
                if (pv.n_candidates === 0) {
                    showBtn.setAttribute("title", "Кандидатов нет — запустите отбор");
                } else {
                    showBtn.removeAttribute("title");
                }
            }
        },

        // ── Модальное окно кандидатов ───────────────────────────────

        /**
         * Открыть модальное окно со списком кандидатов за сегодня.
         */
        openCandidates() {
            const modal = document.getElementById("candidates-modal");
            if (!modal) return;
            modal.hidden = false;
            document.body.classList.add("modal-open");
            const body = document.getElementById("cand-body");
            if (body) body.innerHTML = '<div class="modal-loading"><span class="spinner"></span> Загрузка…</div>';
            fetch("/admin/sources/candidates?limit=200", { headers: { "Accept": "application/json" } })
                .then((r) => r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)))
                .then((data) => adminSources.renderCandidates(data))
                .catch((err) => {
                    const b = document.getElementById("cand-body");
                    if (b) b.innerHTML = '<div class="al-error">Не удалось загрузить: ' +
                        adminSources.esc(err.message) + "</div>";
                });
        },

        /**
         * Закрыть модальное окно. Вызывается по клику на оверлей/кнопку и по Esc.
         * event передаётся из onclick оверлея — закрываем только если клик был
         * именно по оверлею, а не по вложенной карточке.
         */
        closeCandidates(event) {
            if (event && event.target && event.target.closest(".modal-card")) {
                return; // клик внутри карточки — не закрываем
            }
            const modal = document.getElementById("candidates-modal");
            if (modal) modal.hidden = true;
            document.body.classList.remove("modal-open");
        },

        /**
         * Рендер списка кандидатов в тело модального окна.
         */
        renderCandidates(data) {
            const dateEl = document.getElementById("cand-date");
            if (dateEl) dateEl.textContent = data.date || "—";
            // Сводка по классам сложности.
            const sumEl = document.getElementById("cand-summary");
            if (sumEl) {
                const parts = [];
                parts.push("Показано: <strong>" + data.n_shown + "</strong>");
                if (data.n_total != null && data.n_total !== data.n_shown) {
                    parts.push("из " + data.n_total);
                }
                const bd = data.by_difficulty || {};
                const order = ["BOR", "AFC", "SEG", "AGR", "NONE"];
                const chips = order.filter((k) => bd[k]).map((k) =>
                    '<span class="diff-chip diff-' + k.toLowerCase() + '">' + k + ": " + bd[k] + "</span>"
                ).join("");
                sumEl.innerHTML = parts.join(" ") + (chips ? '<span class="diff-chips">' + chips + "</span>" : "");
            }
            // Таблица.
            const body = document.getElementById("cand-body");
            if (!body) return;
            if (!data.candidates || !data.candidates.length) {
                body.innerHTML = '<div class="modal-empty">Нет кандидатов. Запустите шаг 2 (отбор).</div>';
                return;
            }
            const rows = data.candidates.map(function (c) {
                const conf = Math.round((c.confidence || 0) * 100);
                const src = c.source_url
                    ? '<a href="' + adminSources.esc(c.source_url) + '" target="_blank" rel="noopener" title="источник">↗</a>'
                    : "";
                return '<tr>' +
                    '<td class="cand-token">' + adminSources.esc(c.token) + "</td>" +
                    '<td><span class="pos-tag">' + adminSources.esc(c.pos || "—") + "</span></td>" +
                    '<td class="cand-lemma">' + adminSources.esc(c.lemma || "—") + "</td>" +
                    '<td class="num conf-' + (conf < 70 ? "low" : "ok") + '">' + conf + "%</td>" +
                    '<td><span class="diff-badge diff-' + (c.difficulty_class || "none").toLowerCase() + '">' +
                    adminSources.esc(c.difficulty_class || "NONE") + "</span></td>" +
                    '<td class="cand-sentence">' + adminSources.esc(c.sentence || "") + " " + src + "</td>" +
                    "</tr>";
            }).join("");
            body.innerHTML = '<table class="cand-table"><thead><tr>' +
                "<th>Токен</th><th>POS</th><th>Лемма</th><th>Conf</th><th>Сложность</th><th>Предложение</th>" +
                "</tr></thead><tbody>" + rows + "</tbody></table>";
        },

        // ── Утилиты ────────────────────────────────────────────────

        setButtonsDisabled(disabled) {
            document.querySelectorAll(".al-run-btn").forEach(function (b) { b.disabled = disabled; });
        },

        esc(s) {
            return String(s == null ? "" : s)
                .replace(/&/g, "&amp;").replace(/</g, "&lt;")
                .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        },
    };

    // Если страница открыта во время running (F5), подхватываем опрос.
    document.addEventListener("DOMContentLoaded", function () {
        const box = document.getElementById("al-status");
        if (box && box.getAttribute("data-state") === "running") {
            adminSources.setButtonsDisabled(true);
            adminSources.startPolling();
        }
        // Esc закрывает модальное окно кандидатов.
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") adminSources.closeCandidates();
        });
    });

    window.adminSources = adminSources;
})();
