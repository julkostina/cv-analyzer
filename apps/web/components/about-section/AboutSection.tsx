import styles from "./AboutSection.module.css";

export function AboutSection() {
  return (
    <section className={styles.intro} aria-labelledby="about-heading">
      <h1 id="about-heading" className={styles.introTitle}>
        CV Analyzer
      </h1>
      <p className={styles.introLead}>
        This tool helps you see how well your resume fits a job you care about. Upload your CV, optionally add the
        role&apos;s requirements or a link to the posting, and receive structured feedback you can act on.
      </p>
      <ul className={styles.introList}>
        <li>
          <strong>Match insight</strong> — an overall fit score plus a short rationale when a job description is
          provided.
        </li>
        <li>
          <strong>Competencies</strong> — highlights where your profile aligns with the posting and where it may need
          strengthening.
        </li>
        <li>
          <strong>Summary and tips</strong> — strengths, weaknesses, and concrete recommendations.
        </li>
        <li>
          <strong>PDF report</strong> — download a shareable summary of the same analysis.
        </li>
        <li>
          <strong>History</strong> — successful runs are saved in this browser so you can revisit them later (use the
          History link in the header).
        </li>
      </ul>
      <p className={styles.introNote}>
        No account is required. Your files are sent for processing only when you run an analysis or download a report.
      </p>
    </section>
  );
}
