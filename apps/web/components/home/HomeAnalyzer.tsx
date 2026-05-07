"use client";

import { useCallback, useState } from "react";
import { analyzeCv, downloadAnalysisPdf } from "../../lib/analyze-client";
import { addHistoryEntry } from "../../lib/history-storage";
import type { CVAnalysisResponse } from "../../lib/types";
import { triggerBlobDownload } from "../../lib/analysis-format";
import { AnalysisResults } from "./AnalysisResults";
import { AnalyzerForm } from "./AnalyzerForm";
import styles from "./HomeAnalyzer.module.css";

export function HomeAnalyzer() {
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

  return (
    <section className={styles.service} aria-labelledby="service-heading">
      <div className="container">
        <h2 id="service-heading" className={styles.serviceHeading}>
          Run an analysis
        </h2>
        <p className={styles.serviceSub}>Follow the steps below. Supported resume formats: PDF and DOCX.</p>

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

        <AnalysisResults result={result} />
      </div>
    </section>
  );
}
