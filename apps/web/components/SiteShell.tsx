import type { ReactNode } from "react";
import Link from "next/link";
import { uk } from "../lib/strings-uk";
import styles from "./SiteShell.module.css";

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand}>
          {uk.shell.brand}
        </Link>
        <nav className={styles.nav} aria-label={uk.shell.navMain}>
          <Link href="/" className={styles.navLink}>
            {uk.shell.home}
          </Link>
          <Link href="/history" className={styles.navLink}>
            {uk.shell.history}
          </Link>
        </nav>
      </header>
      <div className={styles.content}>{children}</div>
      <footer className={styles.footer}>
        <p>{uk.shell.footer}</p>
      </footer>
    </div>
  );
}
