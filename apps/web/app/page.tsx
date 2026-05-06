"use client";

import { useCallback, useState } from "react";
import { analyzeCv, downloadAnalysisPdf } from "../lib/analyze-client";
import { addHistoryEntry } from "../lib/history-storage";
import type { CVAnalysisResponse } from "../lib/types";
import { getApiBaseUrl } from "../lib/config";
import styles from "./page.module.css";

function formatPercent(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return `${Math.round(score * 1000) / 10}%`;
}

function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CVAnalysisResponse | null>(null);

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setResult(null);
      if (!file) {
        setError("Choose a resume file (PDF or DOCX).");
        return;
      }
      setLoading(true);
      try {
        const data = await analyzeCv(file, {
          jobDescription: jobDescription || undefined,
          jobUrl: jobUrl || undefined,
        });
        setResult(data);
        if (data.success) {
          addHistoryEntry({ fileName: file.name, result: data });
        } else {
          setError(data.error ?? "Analysis finished with an error.");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error.");
      } finally {
        setLoading(false);
      }
    },
    [file, jobDescription, jobUrl],
  );

  const onDownloadPdf = useCallback(async () => {
    if (!file) {
      setError("Choose a resume file first.");
      return;
    }
    setPdfLoading(true);
    setError(null);
    try {
      const blob = await downloadAnalysisPdf(file, {
        jobDescription: jobDescription || undefined,
        jobUrl: jobUrl || undefined,
      });
      triggerBlobDownload(blob, "cv-analysis-report.pdf");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not download PDF.");
    } finally {
      setPdfLoading(false);
    }
  }, [file, jobDescription, jobUrl]);

  const analysis = result?.analysis;
  const summary =
    analysis && typeof analysis.summary === "string" ? analysis.summary : null;
  const strengths = analysis?.strengths;
  const weaknesses = analysis?.weaknesses;

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>Resume vs. job match</h1>
        <p className={styles.subtitle}>
          Upload a resume, add a job description or posting URL, and get a match score and recommendations. API:{" "}
          <code>{getApiBaseUrl()}</code>
        </p>
      </header>

      <form className={styles.steps} onSubmit={onSubmit}>
        <section className={styles.card} aria-labelledby="step1">
          <h2 id="step1" className={styles.cardTitle}>
            Step 1 — resume
          </h2>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="cv-file">
              PDF or DOCX file
            </label>
            <input
              id="cv-file"
              className={styles.input}
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(ev) => {
                const f = ev.target.files?.[0];
                setFile(f ?? null);
              }}
            />
            <span className={styles.hint}>{file ? file.name : "No file selected"}</span>
          </div>
        </section>

        <section className={styles.card} aria-labelledby="step2">
          <h2 id="step2" className={styles.cardTitle}>
            Step 2 — job (optional)
          </h2>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="job-text">
              Job requirements / description
            </label>
            <textarea
              id="job-text"
              className={styles.textarea}
              value={jobDescription}
              onChange={(ev) => setJobDescription(ev.target.value)}
              placeholder="Paste the posting text or key requirements…"
              rows={6}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="job-url">
              Or a link to the job page
            </label>
            <input
              id="job-url"
              className={styles.input}
              type="url"
              inputMode="url"
              value={jobUrl}
              onChange={(ev) => setJobUrl(ev.target.value)}
              placeholder="https://…"
              autoComplete="off"
            />
            <span className={styles.hint}>You can fill only text, only URL, or both.</span>
          </div>
        </section>

        <section className={styles.card} aria-labelledby="step3">
          <h2 id="step3" className={styles.cardTitle}>
            Step 3 — result
          </h2>
          <div className={styles.actions}>
            <button type="submit" className={`${styles.btnBase} ${styles.btnPrimary}`} disabled={loading}>
              {loading ? "Analyzing…" : "Run analysis"}
            </button>
            <button
              type="button"
              className={`${styles.btnBase} ${styles.btnSecondary}`}
              disabled={pdfLoading || !file}
              onClick={onDownloadPdf}
            >
              {pdfLoading ? "PDF…" : "Download PDF"}
            </button>
          </div>
          <p className={styles.hint}>
            The PDF is built on the server with the same request as the analysis; invalid input returns an error
            response instead of a file.
          </p>
        </section>
      </form>

      {error ? (
        <div className={styles.error} role="alert">
          {error}
        </div>
      ) : null}

      {result && !result.success ? (
        <div className={styles.results}>
          {result.extracted_text ? (
            <section className={styles.card} aria-labelledby="partial-heading">
              <h2 id="partial-heading" className={styles.cardTitle}>
                Partial result
              </h2>
              <p className={styles.partialIntro}>
                Text was extracted from the resume file, but structured fields (skills, experience, education, etc.)
                were not filled by the model — see the message above.
              </p>
              <details className={styles.details} open>
                <summary>Extracted resume text</summary>
                <pre className={styles.pre}>{result.extracted_text}</pre>
              </details>
            </section>
          ) : (
            <section className={styles.card}>
              <p className={styles.partialIntro}>
                No extracted text in the response — check API logs and Ollama availability.
              </p>
            </section>
          )}
        </div>
      ) : null}

      {result?.success ? (
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
      ) : null}
    </main>
  );
}
