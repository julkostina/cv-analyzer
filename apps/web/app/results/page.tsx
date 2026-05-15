"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AnalysisResults } from "../../components/home/AnalysisResults";
import { PageHeader } from "../../components/page-header/PageHeader";
import styles from "../../components/home/HomeAnalyzer.module.css";
import { loadLatestResultSession, type LatestResultPayload } from "../../lib/latest-result-session";
import { uk } from "../../lib/strings-uk";
import layoutStyles from "./results.module.css";

export default function ResultsPage() {
  const [payload, setPayload] = useState<LatestResultPayload | null | undefined>(undefined);
  const r = uk.results;

  useEffect(() => {
    setPayload(loadLatestResultSession());
  }, []);

  if (payload === undefined) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader title={r.title} subtitle={r.loading} />
        </main>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className={layoutStyles.page}>
        <main className={layoutStyles.content}>
          <PageHeader
            title={r.title}
            subtitle={
              <>
                {r.emptyBeforeLink}
                <Link href="/">{r.runOnHome}</Link>.
              </>
            }
          />
        </main>
      </div>
    );
  }

  const { result } = payload;

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
