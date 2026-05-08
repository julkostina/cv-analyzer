"use client";

import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { analyzeCv, downloadAnalysisPdf } from "../../lib/analyze-client";
import { addHistoryEntry } from "../../lib/history-storage";
import { saveLatestResultSession } from "../../lib/latest-result-session";
import type { CVAnalysisResponse } from "../../lib/types";
import { triggerBlobDownload } from "../../lib/analysis-format";
import { AnalysisResults } from "./AnalysisResults";
import { AnalyzerForm } from "./AnalyzerForm";
import styles from "./HomeAnalyzer.module.css";

export function HomeAnalyzer() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inlineResult, setInlineResult] = useState<CVAnalysisResponse | null>(null);

  const onSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);
      setInlineResult(null);
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
        const saved = saveLatestResultSession(file.name, data);
        if (data.success) {
          addHistoryEntry({ fileName: file.name, result: data });
        }
        if (!saved) {
          setInlineResult(data);
          setError(
            "Could not open the results page: this browser blocked or ran out of session storage. Your analysis is shown below — allow storage for this site and run again to use the results page.",
          );
          return;
        }
        router.push("/results");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error.");
      } finally {
        setLoading(false);
      }
    },
    [file, jobDescription, jobUrl, router],
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

  return (
    <section className={styles.service} aria-labelledby="service-heading">
      <div className="container">
        <h2 id="service-heading" className={styles.serviceHeading}>
          Analyze your resume
        </h2>
        <p className={styles.serviceSub}>
          Supported formats: <strong>PDF</strong> and <strong>DOCX</strong>.
        </p>

        <AnalyzerForm
          file={file}
          onFileChange={setFile}
          jobDescription={jobDescription}
          onJobDescriptionChange={setJobDescription}
          jobUrl={jobUrl}
          onJobUrlChange={setJobUrl}
          loading={loading}
          pdfLoading={pdfLoading}
          onSubmit={onSubmit}
          onDownloadPdf={onDownloadPdf}
        />

        {error ? (
          <div className={styles.error} role="alert">
            {error}
          </div>
        ) : null}

        {inlineResult ? <AnalysisResults result={inlineResult} /> : null}
      </div>
    </section>
  );
}
