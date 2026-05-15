import { formatPercent, matchScoreUnits } from "../../lib/analysis-format";
import { gaugeAriaLabel, uk } from "../../lib/strings-uk";
import type { CVAnalysisResponse, ExplainabilityMethodResult, MatchExplainability } from "../../lib/types";
import styles from "./HomeAnalyzer.module.css";

type AnalysisResultsProps = {
  result: CVAnalysisResponse | null;
};

const u = uk.analysisResults;

/** Тримайте синхронно з `apps/api/app/services/semantic_matcher.py` `SEMANTIC_METRIC_GUIDES` (кешовані сесії можуть не мати копії з API). */
const FALLBACK_SEMANTIC_GUIDES: Record<string, string> = { ...u.fallbackGuides };

function formatSemanticWeight(n: number | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return (Math.round(n * 1000) / 1000).toString();
}

/** Верхня півкола: плоска хорда внизу, дуга відкривається вгору (стиль спідометра). */
const GAUGE_CX = 50;
const GAUGE_CY = 46;
const GAUGE_R = 38;
const GAUGE_PATH = `M ${GAUGE_CX - GAUGE_R} ${GAUGE_CY} A ${GAUGE_R} ${GAUGE_R} 0 0 0 ${GAUGE_CX + GAUGE_R} ${GAUGE_CY}`;

function JobMatchGaugeSvg({ progress }: { progress: number }) {
  const t = Math.max(0, Math.min(1, progress));
  const dashProgress = `${t * 1000} 1000`;

  const strokeW = 12;

  return (
    <svg className={styles.matchGaugeSvg} viewBox="0 0 100 52" role="presentation" aria-hidden>
      <path
        className={styles.matchGaugeTrackRim}
        d={GAUGE_PATH}
        fill="none"
        strokeWidth={strokeW}
        strokeLinecap="butt"
        pathLength={1000}
      />
      <path
        className={styles.matchGaugeArc}
        d={GAUGE_PATH}
        fill="none"
        strokeWidth={strokeW}
        strokeLinecap="butt"
        pathLength={1000}
        strokeDasharray={dashProgress}
      />
    </svg>
  );
}

type SemanticBreakdown = NonNullable<CVAnalysisResponse["semantic_breakdown"]>;

function JobMatchSemanticRow({ breakdown }: { breakdown: SemanticBreakdown }) {
  return (
    <div className={styles.semanticRow}>
      <div className={styles.semanticMetric}>
        <span className={`${styles.semanticSwatch} ${styles.semanticSwatchSkills}`} aria-hidden />
        <span>
          {u.semanticRow.skills}: {formatPercent(breakdown.skills_similarity ?? 0)}
        </span>
      </div>
      <span className={styles.semanticSep} aria-hidden>
        ·
      </span>
      <div className={styles.semanticMetric}>
        <span className={`${styles.semanticSwatch} ${styles.semanticSwatchExperience}`} aria-hidden />
        <span>
          {u.semanticRow.experience}: {formatPercent(breakdown.experience_similarity ?? 0)}
        </span>
      </div>
      <span className={styles.semanticSep} aria-hidden>
        ·
      </span>
      <div className={styles.semanticMetric}>
        <span className={`${styles.semanticSwatch} ${styles.semanticSwatchOverall}`} aria-hidden />
        <span>
          {u.semanticRow.overall}: {formatPercent(breakdown.overall_similarity ?? 0)}
        </span>
      </div>
    </div>
  );
}

const SEMANTIC_GUIDE_ORDER = ["skills", "experience", "overall", "match_score"] as const;

const SEMANTIC_GUIDE_TITLES: Record<(typeof SEMANTIC_GUIDE_ORDER)[number], string> = {
  skills: u.guideTitles.skills,
  experience: u.guideTitles.experience,
  overall: u.guideTitles.overall,
  match_score: u.guideTitles.match_score,
};

function JobMatchBlock({
  score,
  breakdown,
  reasoning,
  weights,
  guides,
  narrative,
  explainability,
}: {
  score: number;
  breakdown: CVAnalysisResponse["semantic_breakdown"];
  reasoning: CVAnalysisResponse["match_score_reasoning"];
  weights: CVAnalysisResponse["semantic_weights"];
  guides: CVAnalysisResponse["semantic_metric_guides"];
  narrative: CVAnalysisResponse["semantic_score_narrative"];
  explainability: CVAnalysisResponse["match_explainability"];
}) {
  const units = matchScoreUnits(score);
  if (units == null) return null;

  const wSkills = weights?.skills ?? 0.5;
  const wExp = weights?.experience ?? 0.3;
  const wOverall = weights?.overall ?? 0.2;
  const guideText = breakdown ? guides ?? FALLBACK_SEMANTIC_GUIDES : null;

  return (
    <div className={styles.matchBlock}>
      <figure className={styles.matchGaugeFigure} aria-label={gaugeAriaLabel(units)}>
        <JobMatchGaugeSvg progress={score} />
      </figure>
      <div className={styles.matchScoreStack}>
        <span className={styles.matchScoreUnits}>{units}</span>
        <span className={styles.matchScoreSub}>{u.outOf100}</span>
      </div>
      {breakdown ? (
        <>
          <JobMatchSemanticRow breakdown={breakdown} />
          <p className={styles.semanticWeightsLine}>
            {u.weightsLine(formatSemanticWeight(wSkills), formatSemanticWeight(wExp), formatSemanticWeight(wOverall))}
          </p>
          {narrative?.trim() ? (
            <div className={styles.semanticNarrative}>
              <h3 className={styles.semanticNarrativeTitle}>{u.narrativeTitle}</h3>
              {narrative
                .trim()
                .split(/\n\s*\n/)
                .filter(Boolean)
                .map((para, i) => (
                  <p key={i} className={styles.semanticNarrativePara}>
                    {para.trim()}
                  </p>
                ))}
            </div>
          ) : null}
          {guideText ? (
            <details className={styles.semanticHelp}>
              <summary>{u.semanticHelpSummary}</summary>
              <ol className={styles.semanticHelpList}>
                {SEMANTIC_GUIDE_ORDER.map((key) => {
                  const body = guideText[key];
                  if (!body) return null;
                  return (
                    <li key={key}>
                      <strong>{SEMANTIC_GUIDE_TITLES[key]}</strong> — {body}
                    </li>
                  );
                })}
              </ol>
            </details>
          ) : null}
        </>
      ) : null}
      {reasoning ? <p className={styles.reasoning}>{reasoning}</p> : null}
      {explainability ? <MatchExplainabilityBlock data={explainability} /> : null}
    </div>
  );
}

const COMPONENT_ORDER = ["skills", "experience", "overall"] as const;

function MatchExplainabilityBlock({ data }: { data: MatchExplainability }) {
  const comps = data.component_attributions;
  const hasComponent = comps && COMPONENT_ORDER.some((k) => comps[k] != null);
  const hasShap = data.shap && (data.shap.top_positive.length > 0 || data.shap.top_negative.length > 0);
  const hasLime = data.lime && (data.lime.top_positive.length > 0 || data.lime.top_negative.length > 0);
  if (!hasComponent && !hasShap && !hasLime) return null;

  return (
    <details className={styles.explainability}>
      <summary>{u.explainabilityTitle}</summary>
      <p className={styles.explainabilityIntro}>{u.explainabilityIntro}</p>
      {hasComponent ? (
        <div className={styles.explainabilitySection}>
          <h3 className={styles.explainabilitySubtitle}>{u.componentAttributionsTitle}</h3>
          <ul className={styles.explainabilityList}>
            {COMPONENT_ORDER.map((key) => {
              const value = comps?.[key];
              if (value == null) return null;
              return (
                <li key={key}>
                  {u.componentLabels[key]}: {u.contributionValue(value)}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
      {hasShap ? <ExplainabilityMethodBlock title={u.shapTitle} method={data.shap!} /> : null}
      {hasLime ? <ExplainabilityMethodBlock title={u.limeTitle} method={data.lime!} /> : null}
    </details>
  );
}

function ExplainabilityMethodBlock({ title, method }: { title: string; method: ExplainabilityMethodResult }) {
  return (
    <div className={styles.explainabilitySection}>
      <h3 className={styles.explainabilitySubtitle}>{title}</h3>
      {method.top_positive.length > 0 ? (
        <>
          <p className={styles.explainabilityLabel}>{u.raisesScore}</p>
          <ul className={styles.explainabilityList}>
            {method.top_positive.map((item, i) => (
              <li key={`p-${i}`}>{u.contributionLine(item.feature, item.contribution)}</li>
            ))}
          </ul>
        </>
      ) : null}
      {method.top_negative.length > 0 ? (
        <>
          <p className={`${styles.explainabilityLabel} ${styles.sectionSpaced}`}>{u.lowersScore}</p>
          <ul className={styles.explainabilityList}>
            {method.top_negative.map((item, i) => (
              <li key={`n-${i}`}>{u.contributionLine(item.feature, item.contribution)}</li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

export function AnalysisResults({ result }: AnalysisResultsProps) {
  if (!result) return null;

  if (!result.success) {
    return (
      <div className={styles.results}>
        {result.extracted_text ? (
          <section className={styles.card} aria-labelledby="partial-heading">
            <h2 id="partial-heading" className={styles.cardTitle}>
              {u.partialTitle}
            </h2>
            <p className={styles.partialIntro}>{u.partialIntro}</p>
            <details className={styles.details} open>
              <summary>{u.extractedSummary}</summary>
              <pre className={styles.pre}>{result.extracted_text}</pre>
            </details>
          </section>
        ) : (
          <section className={styles.card}>
            <p className={styles.partialIntro}>{u.partialEmpty}</p>
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
        <section className={`${styles.card} ${styles.cardJobMatch}`}>
          <h2 className={styles.cardTitle}>{u.jobMatch}</h2>
          <JobMatchBlock
            score={result.match_score}
            breakdown={result.semantic_breakdown}
            reasoning={result.match_score_reasoning}
            weights={result.semantic_weights}
            guides={result.semantic_metric_guides}
            narrative={result.semantic_score_narrative}
            explainability={result.match_explainability}
          />
        </section>
      ) : null}

      {(result.matched_competencies?.length ?? 0) > 0 || (result.missing_competencies?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>{u.competencies}</h2>
          {(result.matched_competencies?.length ?? 0) > 0 ? (
            <>
              <p className={styles.label}>{u.aligned}</p>
              <ul className={styles.list}>
                {result.matched_competencies!.map((x, i) => (
                  <li key={`m-${i}`}>{x}</li>
                ))}
              </ul>
            </>
          ) : null}
          {(result.missing_competencies?.length ?? 0) > 0 ? (
            <>
              <p className={`${styles.label} ${styles.sectionSpaced}`}>{u.needsAttention}</p>
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
          <h2 className={styles.cardTitle}>{u.analysisSummary}</h2>
          {summary ? <p className={styles.reasoning}>{summary}</p> : null}
          {strengths ? (
            <>
              <p className={`${styles.label} ${styles.sectionSpaced}`}>{u.strengths}</p>
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
              <p className={`${styles.label} ${styles.sectionSpaced}`}>{u.weaknesses}</p>
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
          <h2 className={styles.cardTitle}>{u.recommendations}</h2>
          <ul className={styles.list}>
            {result.recommendations!.map((r, i) => (
              <li key={`r-${i}`}>{r}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {(result.skills?.length ?? 0) > 0 ? (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>{u.skillsFromResume}</h2>
          <p className={styles.reasoning}>{result.skills!.join(", ")}</p>
        </section>
      ) : null}

      {result.extracted_text ? (
        <details className={styles.details}>
          <summary>{u.extractedSummary}</summary>
          <pre className={styles.pre}>{result.extracted_text}</pre>
        </details>
      ) : null}
    </div>
  );
}
