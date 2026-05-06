import styles from "./HomeAnalyzer.module.css";

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
              onFileChange(f ?? null);
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
            onChange={(ev) => onJobDescriptionChange(ev.target.value)}
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
            onChange={(ev) => onJobUrlChange(ev.target.value)}
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
          The PDF reflects the same inputs as your analysis. If the resume or job details cannot be processed, you
          will see an error message instead of a download.
        </p>
      </section>
    </form>
  );
}
