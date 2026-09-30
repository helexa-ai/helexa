import { useTranslation } from "react-i18next";
import type { Draft } from "../../lib/decisionRequest";
import { EXAMPLES, EXAMPLE_GROUPS, exampleDraft } from "../../data/decisionExamples";
import "./examples.css";

/**
 * Lessons, use cases and the multilingual set. Picking one fills the
 * editors; nothing is sent until Run.
 */
export default function ExamplesPanel({ onPick }: { onPick: (draft: Draft) => void }) {
  const { t } = useTranslation("decisions");
  return (
    <div className="hx-pg-examples">
      <p className="hx-pg-help">{t("examples.intro")}</p>
      {EXAMPLE_GROUPS.map((g) => (
        <section key={g} aria-label={t(`examples.groups.${g}`)}>
          <h3 className="hx-pg-eyebrow">{t(`examples.groups.${g}`)}</h3>
          <div className={g === "lessons" ? "hx-pg-lessons" : "hx-pg-cases"}>
            {EXAMPLES.filter((e) => e.group === g).map((ex) => (
              <button
                key={ex.id}
                type="button"
                className={g === "lessons" ? "hx-pg-lesson" : "hx-pg-case"}
                onClick={() => onPick(exampleDraft(ex))}
              >
                {ex.primitive && <span className={`hx-pg-type hx-pg-type-${ex.primitive}`}>{ex.primitive}</span>}
                <strong>{t(`examples.${ex.id}.title`)}</strong>
                <span className="hx-pg-help">{t(`examples.${ex.id}.blurb`)}</span>
              </button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
