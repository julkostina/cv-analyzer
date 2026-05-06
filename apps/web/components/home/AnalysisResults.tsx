import { formatPercent } from "../../lib/analysis-format";
import type { CVAnalysisResponse } from "../../lib/types";
import styles from "./HomeAnalyzer.module.css";

type AnalysisResultsProps = {
  result: CVAnalysisResponse | null;
};

export function AnalysisResults({ result }: AnalysisResultsProps) {
  if (!result) return null;

  if (!result.success) {
    return (
      <div className={styles.results}>
        {result.extracted_text ? (
          <section className={styles.card} aria-labelledby="partial-heading">
            <h2 id="partial-heading" className={styles.cardTitle}>
              Partial result
            </h2>
            <p className={styles.partialIntro}>
              Text was extracted from the resume file, but structured fields (skills, experience, education, etc.) were
              not filled by the model — see the message above.
            </p>
            <details className={styles.details} open>
              <summary>Extracted resume text</summary>
              <pre className={styles.pre}>{result.extracted_text}</pre>
            </details>
          </section>
        ) : (
          <section className={styles.card}>
            <p className={styles.partialIntro}>
              No resume text was returned. Try again with another file or check that the analysis service is running.
            </p>
          </section>
        )}
      </div>
    );
  }

  const analysis = result.analysis;
  const summary = analysis && typeof analysis.summary === "string" ? analysis.summary : null;
  const strengths = analysis?.strengths;
  const weaknesses = analysis?.weaknesses;

  return (
    <div className={styles.results}>
      {result.match_score != null ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Job match</h2>
          <div className={styles.scoreRow}>
            <span className={styles.scoreBig}>{formatPercent(result.match_score)}</span>
            <span className={styles.hint}>semantic score (0–100%)</span>
          </div>
          {result.semantic_breakdown ? (
            <div className={styles.semantic}>
              <span>
                Skills: {formatPercent(result.semantic_breakdown.skills_similarity ?? 0)} · Experience:{" "}
                {formatPercent(result.semantic_breakdown.experience_similarity ?? 0)} · Overall:{" "}
                {formatPercent(result.semantic_breakdown.overall_similarity ?? 0)}
              </span>
            </div>
          ) : null}
          {result.match_score_reasoning ? (
            <p className={styles.reasoning}>{result.match_score_reasoning}</p>
          ) : null}
        </section>
      ) : null}

      {(result.matched_competencies?.length ?? 0) > 0 || (result.missing_competencies?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Competencies</h2>
          {(result.matched_competencies?.length ?? 0) > 0 ? (
            <>
              <p className={styles.label}>Aligned with requirements</p>
              <ul className={styles.list}>
                {result.matched_competencies!.map((x, i) => (
                  <li key={`m-${i}`}>{x}</li>
                ))}
              </ul>
            </>
          ) : null}
          {(result.missing_competencies?.length ?? 0) > 0 ? (
            <>
              <p className={`${styles.label} ${styles.sectionSpaced}`}>Needs attention</p>
              <ul className={styles.list}>
                {result.missing_competencies!.map((x, i) => (
                  <li key={`g-${i}`}>{x}</li>
                ))}
              </ul>
            </>
          ) : null}
        </section>
      ) : null}

      {summary || strengths || weaknesses ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Analysis summary</h2>
          {summary ? <p className={styles.reasoning}>{summary}</p> : null}
          {strengths ? (
            <>
              <p className={`${styles.label} ${styles.sectionSpaced}`}>Strengths</p>
              {Array.isArray(strengths) ? (
                <ul className={styles.list}>
                  {strengths.map((s, i) => (
                    <li key={`s-${i}`}>{String(s)}</li>
                  ))}
                </ul>
              ) : (
                <p className={styles.reasoning}>{String(strengths)}</p>
              )}
            </>
          ) : null}
          {weaknesses ? (
            <>
              <p className={`${styles.label} ${styles.sectionSpaced}`}>Weaknesses</p>
              {Array.isArray(weaknesses) ? (
                <ul className={styles.list}>
                  {weaknesses.map((w, i) => (
                    <li key={`w-${i}`}>{String(w)}</li>
                  ))}
                </ul>
              ) : (
                <p className={styles.reasoning}>{String(weaknesses)}</p>
              )}
            </>
          ) : null}
        </section>
      ) : null}

      {(result.recommendations?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Recommendations</h2>
          <ul className={styles.list}>
            {result.recommendations!.map((r, i) => (
              <li key={`r-${i}`}>{r}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {(result.skills?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Skills from resume</h2>
          <p className={styles.reasoning}>{result.skills!.join(", ")}</p>
        </section>
      ) : null}

      {result.extracted_text ? (
        <details className={styles.details}>
          <summary>Extracted resume text</summary>
          <pre className={styles.pre}>{result.extracted_text}</pre>
        </details>
      ) : null}
    </div>
  );
}
