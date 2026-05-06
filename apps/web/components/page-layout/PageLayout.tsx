import type { ReactNode } from "react";
import styles from "./PageLayout.module.css";

export function PageLayout({ children }: { children: ReactNode }) {
  return <main className={styles.main}>{children}</main>;
}
