import type { ReactNode } from "react";
import styles from "./PageHeader.module.css";

type PageHeaderProps = {
  title: string;
  subtitle?: ReactNode;
};

export function PageHeader({ title, subtitle }: PageHeaderProps) {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>{title}</h1>
      {subtitle ? <div className={styles.subtitle}>{subtitle}</div> : null}
    </header>
  );
}
