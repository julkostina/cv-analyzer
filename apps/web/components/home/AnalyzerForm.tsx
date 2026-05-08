import styles from "./HomeAnalyzer.module.css";

const DOWNLOAD_PDF_TOOLTIP =
  "The PDF reflects the same inputs as your analysis. If the resume or job details cannot be processed, you will see an error message instead of a download.";

export type AnalyzerFormProps = {
  file: File | null;
  onFileChange: (file: File | null) => void;
  jobDescription: string;
  onJobDescriptionChange: (value: string) => void;
  jobUrl: string;
  onJobUrlChange: (value: string) => void;
  loading: boolean;
  pdfLoading: boolean;
  onSubmit: (e: React.FormEvent) => void;
  onDownloadPdf: () => void;
};

export function AnalyzerForm({
  file,
  onFileChange,
  jobDescription,
  onJobDescriptionChange,
  jobUrl,
  onJobUrlChange,
  loading,
  pdfLoading,
  onSubmit,
  onDownloadPdf,
}: AnalyzerFormProps) {
  return (
    <form className={styles.steps} onSubmit={onSubmit}>
      <section className={styles.card} aria-labelledby="resume-heading">
        <h2 id="resume-heading" className={styles.cardTitle}>
          Resume
        </h2>
        <div className={styles.resumeRow}>
          <div className={`${styles.field} ${styles.resumeFileField}`}>
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
                onFileChange(f ?? null);
              }}
            />
          </div>
          <div className={`${styles.field} ${styles.resumeUrlField}`}>
            <label className={styles.label} htmlFor="job-url">
              Link to the job page
            </label>
            <input
              id="job-url"
              className={styles.input}
              type="url"
              inputMode="url"
              value={jobUrl}
              onChange={(ev) => onJobUrlChange(ev.target.value)}
              placeholder="https://…"
              autoComplete="off"
            />
          </div>
        </div>
      </section>

      <section className={styles.card} aria-labelledby="job-heading">
        <h2 id="job-heading" className={styles.cardTitle}>
          Job (optional)
        </h2>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="job-text">
            Job requirements / description
          </label>
          <textarea
            id="job-text"
            className={styles.textarea}
            value={jobDescription}
            onChange={(ev) => onJobDescriptionChange(ev.target.value)}
            placeholder="Paste the posting text or key requirements…"
            rows={6}
          />
        </div>
      </section>

      <section className={styles.card} aria-label="Run analysis or download PDF">
        <div className={styles.actions}>
          <button type="submit" className={`${styles.btnBase} ${styles.btnPrimary}`} disabled={loading}>
            {loading ? "Analyzing…" : "Run analysis"}
          </button>
          <button
            type="button"
            className={`${styles.btnBase} ${styles.btnSecondary}`}
            disabled={pdfLoading || !file}
            onClick={onDownloadPdf}
            title={DOWNLOAD_PDF_TOOLTIP}
            aria-describedby="download-pdf-help"
          >
            {pdfLoading ? "PDF…" : "Download PDF"}
          </button>
        </div>
        <span id="download-pdf-help" className={styles.visuallyHidden}>
          {DOWNLOAD_PDF_TOOLTIP}
        </span>
      </section>
    </form>
  );
}
