"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AnalysisResults } from "../../components/home/AnalysisResults";
import { PageHeader } from "../../components/page-header/PageHeader";
import { PageLayout } from "../../components/page-layout/PageLayout";
import styles from "../../components/home/HomeAnalyzer.module.css";
import { loadLatestResultSession, type LatestResultPayload } from "../../lib/latest-result-session";

export default function ResultsPage() {
  const [payload, setPayload] = useState<LatestResultPayload | null | undefined>(undefined);

  useEffect(() => {
    setPayload(loadLatestResultSession());
  }, []);

  if (payload === undefined) {
    return (
      <PageLayout>
        <PageHeader title="Job match results" subtitle="Loading…" />
      </PageLayout>
    );
  }

  if (!payload) {
    return (
      <PageLayout>
        <PageHeader
          title="Job match results"
          subtitle={
            <>
              No analysis in this session yet. <Link href="/">Run an analysis on the home page</Link>.
            </>
          }
        />
      </PageLayout>
    );
  }

  const { result, fileName } = payload;

  return (
    <PageLayout>
      <PageHeader
        title="Job match results"
        subtitle={
          <>
            <strong>{fileName}</strong>
            {" · "}
            <Link href="/">Back to analyzer</Link>
            {" · "}
            <Link href="/history">History</Link>
          </>
        }
      />

      {result.success === false && result.error ? (
        <div className={styles.error} role="alert">
          {result.error}
        </div>
      ) : null}

      <AnalysisResults result={result} />
    </PageLayout>
  );
}
