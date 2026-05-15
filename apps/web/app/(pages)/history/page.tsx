"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  clearHistory,
  loadHistory,
  removeHistoryEntry,
  type HistoryEntry,
} from "../../../lib/history-storage";
import { formatPercent } from "../../../lib/analysis-format";
import { uk } from "../../../lib/strings-uk";
import { PageHeader } from "../../../components/page-header/PageHeader";
import { PageLayout } from "../../../components/page-layout/PageLayout";
import surface from "../../../components/ui/surface.module.css";
import historyStyles from "./history.module.css";

function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat("uk-UA", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const h = uk.history;

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
    <PageLayout>
      <PageHeader
        title={h.title}
        subtitle={
          <>
            {h.subtitleBeforeLink}
            <Link href="/">{h.backToAnalyzer}</Link>
          </>
        }
      />

      {entries.length === 0 ? (
        <section className={surface.card} aria-labelledby="empty-history">
          <h2 id="empty-history" className={surface.cardTitle}>
            {h.emptyTitle}
          </h2>
          <p className={surface.reasoning}>{h.emptyBody}</p>
        </section>
      ) : (
        <>
          <div className={historyStyles.toolbar}>
            <p className={historyStyles.count}>{h.count(entries.length)}</p>
            <button type="button" className={historyStyles.dangerBtn} onClick={onClear}>
              {h.clearAll}
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
                    <span>
                      {h.match}: {formatPercent(e.result.match_score)}
                    </span>
                  ) : (
                    <span>{h.matchDash}</span>
                  )}
                  {(e.result.recommendations?.length ?? 0) > 0 ? (
                    <span>{h.tips(e.result.recommendations!.length)}</span>
                  ) : null}
                </div>
                {e.result.analysis && typeof e.result.analysis.summary === "string" ? (
                  <p className={historyStyles.summary}>{e.result.analysis.summary}</p>
                ) : null}
                <button type="button" className={historyStyles.removeBtn} onClick={() => onRemove(e.id)}>
                  {h.remove}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </PageLayout>
  );
}
