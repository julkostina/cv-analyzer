"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  clearHistory,
  loadHistory,
  removeHistoryEntry,
  type HistoryEntry,
} from "../../lib/history-storage";
import styles from "../page.module.css";
import historyStyles from "./history.module.css";

function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function formatPercent(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return `${Math.round(score * 1000) / 10}%`;
}

export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  const refresh = useCallback(() => {
    setEntries(loadHistory());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onRemove = useCallback(
    (id: string) => {
      removeHistoryEntry(id);
      refresh();
    },
    [refresh],
  );

  const onClear = useCallback(() => {
    clearHistory();
    refresh();
  }, [refresh]);

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>Analysis history</h1>
        <p className={styles.subtitle}>
          Successful runs are saved in this browser. Clearing site data removes them.{" "}
          <Link href="/">Back to analyzer</Link>
        </p>
      </header>

      {entries.length === 0 ? (
        <section className={styles.card} aria-labelledby="empty-history">
          <h2 id="empty-history" className={styles.cardTitle}>
            No saved analyses yet
          </h2>
          <p className={styles.reasoning}>
            Run an analysis on the home page. When it succeeds, it is added here automatically.
          </p>
        </section>
      ) : (
        <>
          <div className={historyStyles.toolbar}>
            <p className={historyStyles.count}>
              {entries.length} {entries.length === 1 ? "entry" : "entries"}
            </p>
            <button type="button" className={historyStyles.dangerBtn} onClick={onClear}>
              Clear all
            </button>
          </div>
          <ul className={historyStyles.list}>
            {entries.map((e) => (
              <li key={e.id} className={historyStyles.item}>
                <div className={historyStyles.itemHead}>
                  <span className={historyStyles.fileName}>{e.fileName}</span>
                  <time className={historyStyles.time} dateTime={e.savedAt}>
                    {formatDate(e.savedAt)}
                  </time>
                </div>
                <div className={historyStyles.meta}>
                  {e.result.match_score != null ? (
                    <span>Match: {formatPercent(e.result.match_score)}</span>
                  ) : (
                    <span>Match: —</span>
                  )}
                  {(e.result.recommendations?.length ?? 0) > 0 ? (
                    <span>{e.result.recommendations!.length} recommendations</span>
                  ) : null}
                </div>
                {e.result.analysis && typeof e.result.analysis.summary === "string" ? (
                  <p className={historyStyles.summary}>{e.result.analysis.summary}</p>
                ) : null}
                <button type="button" className={historyStyles.removeBtn} onClick={() => onRemove(e.id)}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </main>
  );
}
