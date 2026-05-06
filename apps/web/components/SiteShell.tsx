import type { ReactNode } from "react";
import Link from "next/link";
import styles from "./SiteShell.module.css";

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand}>
          CV Analyzer
        </Link>
        <nav className={styles.nav} aria-label="Main">
          <Link href="/" className={styles.navLink}>
            Home
          </Link>
          <Link href="/history" className={styles.navLink}>
            History
          </Link>
        </nav>
      </header>
      <div className={styles.content}>{children}</div>
      <footer className={styles.footer}>
        <p>No sign-up. History of successful analyses is kept in this browser only and is not sent anywhere else.</p>
      </footer>
    </div>
  );
}
