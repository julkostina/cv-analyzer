"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AnalysisResults } from "../../components/home/AnalysisResults";
import { PageHeader } from "../../components/page-header/PageHeader";
import styles from "../../components/home/HomeAnalyzer.module.css";
import { loadLatestResultSession, type LatestResultPayload } from "../../lib/latest-result-session";
import layoutStyles from "./results.module.css";

export default function ResultsPage() {
  const [payload, setPayload] = useState<LatestResultPayload | null | undefined>(undefined);

  useEffect(() => {
    setPayload(loadLatestResultSession());
  }, []);

  if (payload === undefined) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader title="Job match results" subtitle="Loading…" />
        </main>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader
            title="Job match results"
            subtitle={
              <>
                No analysis in this session yet. <Link href="/">Run an analysis on the home page</Link>.
              </>
            }
          />
        </main>
      </div>
    );
  }

  const { result, fileName } = payload;

  return (
    <div className={layoutStyles.page}>
      <main className={layoutStyles.content}>
        {result.success === false && result.error ? (
          <div className={styles.error} role="alert">
            {result.error}
          </div>
        ) : null}

        <AnalysisResults result={result} />
      </main>
    </div>
  );
}
