/* Admin review: AJAX handlers for approve/reject/edit decisions.
   Posts to /admin/review/{id}/decision, updates the card UI in place. */
(function () {
    "use strict";

    const adminReview = {
        /**
         * Submit a decision (approved/rejected/edited) for an annotation.
         * On success, replaces the action buttons with the final result.
         */
        decide(annotationId, decision, extra) {
            const body = Object.assign({ decision: decision }, extra || {});
            fetch(`/admin/review/${annotationId}/decision`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            })
                .then((r) => {
                    if (!r.ok) throw new Error("HTTP " + r.status);
                    return r.json();
                })
                .then((data) => {
                    if (data.ok) {
                        adminReview.markReviewed(annotationId, data);
                    } else {
                        alert("Ошибка: " + (data.detail || "неизвестная"));
                    }
                })
                .catch((err) => alert("Не удалось сохранить: " + err.message));
        },

        /**
         * Toggle the inline edit form for an annotation.
         */
        toggleEdit(annotationId) {
            const form = document.getElementById("edit-form-" + annotationId);
            if (form) {
                form.style.display = form.style.display === "none" ? "flex" : "none";
            }
        },

        /**
         * Submit the edited lemma/pos as a human override.
         */
        submitEdit(annotationId) {
            const lemma = document.getElementById("edit-lemma-" + annotationId).value.trim();
            const pos = document.getElementById("edit-pos-" + annotationId).value.trim().toUpperCase();
            if (!lemma || !pos) {
                alert("Нужны и лемма, и POS для edited.");
                return;
            }
            adminReview.decide(annotationId, "edited", {
                final_lemma: lemma,
                final_pos: pos,
            });
        },

        /**
         * Replace the action area with the final result after a decision.
         */
        markReviewed(annotationId, data) {
            const actions = document.querySelector(
                `.review-actions[data-annotation-id="${annotationId}"]`
            );
            const editForm = document.getElementById("edit-form-" + annotationId);
            if (editForm) editForm.remove();
            if (actions) {
                const status = data.status || "approved";
                const finalLemma = data.final_lemma || "—";
                const finalPos = data.final_pos || "—";
                actions.outerHTML =
                    `<div class="review-final">Итог: ` +
                    `<span class="pos-tag">${finalPos}</span> ` +
                    `<span class="lemma">${finalLemma}</span> ` +
                    `<span class="status-badge status-${status}">${status}</span></div>`;
            }
            // Header badge update: show status.
            const header = document.querySelector(`#ann-${annotationId} .review-card-header`);
            if (header && !header.querySelector(".status-badge")) {
                const badge = document.createElement("span");
                badge.className = "status-badge status-" + (data.status || "approved");
                badge.textContent = data.status || "approved";
                header.appendChild(badge);
            }
        },
    };

    window.adminReview = adminReview;
})();
