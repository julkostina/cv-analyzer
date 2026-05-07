"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useLayoutEffect, useRef, useState } from "react";
import styles from "./AboutSection.module.css";

const FEATURES = [
  {
    title: "Fit score",
    description:
      "An overall match percentage with a short rationale explaining what's driving it.",
  },
  {
    title: "Competency breakdown",
    description:
      "What aligns with the posting and what may need strengthening, section by section.",
  },
  {
    title: "Strengths & recommendations",
    description:
      "A plain-language summary of what's working and concrete suggestions for what to change.",
  },
  {
    title: "PDF report",
    description: "A downloadable summary you can save or share.",
  },
  {
    title: "History",
    description:
      "Past analyses are stored in this browser so you can track changes across versions of your CV.",
  },
] as const;

function FeatureCard({ title, description }: (typeof FEATURES)[number]) {
  return (
    <article className={styles.featureCard}>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
    </article>
  );
}

function FeatureMarqueeStatic() {
  return (
    <div className="container">
      <div className={styles.carouselReduced}>
        <ul className={styles.carouselReducedList}>
          {FEATURES.map((feature) => (
            <li key={feature.title} className={styles.carouselReducedItem}>
              <FeatureCard {...feature} />
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function FeatureMarqueeAnimated() {
  const trackRef = useRef<HTMLDivElement>(null);
  const [loopWidth, setLoopWidth] = useState(0);

  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const measure = () => {
      const w = el.scrollWidth;
      setLoopWidth(w > 0 ? w / 2 : 0);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const duration = loopWidth > 0 ? Math.max(22, loopWidth / 42) : 0;

  return (
    <div className={styles.carouselViewport} role="region" aria-label="What you get: feature highlights">
      <motion.div
        ref={trackRef}
        className={styles.carouselTrack}
        animate={loopWidth > 0 ? { x: [0, -loopWidth] } : undefined}
        transition={{
          duration,
          repeat: Infinity,
          ease: "linear",
        }}
      >
        {[...FEATURES, ...FEATURES].map((feature, index) => (
          <div key={`${feature.title}-${index}`} className={styles.carouselCardWrap}>
            <FeatureCard {...feature} />
          </div>
        ))}
      </motion.div>
    </div>
  );
}

export function AboutSection() {
  const prefersReducedMotion = useReducedMotion() === true;

  return (
    <>
      <section className={styles.intro} aria-labelledby="about-heading">
        <div className={`container ${styles.introContent}`}>
          <h1 id="about-heading" className={styles.introTitle}>
            CV Analyzer
          </h1>
          <p className={styles.introLead}>
            You send a resume, hear nothing, and have no idea whether the problem was your experience, your wording, or
            something an algorithm flagged before anyone read a word.
          </p>
          <p className={styles.introLead}>
            This tool gives you a way to check before you apply. Upload your CV, add a job description or a link to the
            posting, and get structured feedback tied to that specific role — not generic advice, but an analysis of how
            your profile actually reads against what the employer is asking for.
          </p>
          <p className={styles.introNote}>
            No account needed. Your files are used only during analysis and never stored after processing completes.
          </p>
        </div>
      </section>

      <section className={styles.featuresSection} aria-labelledby="about-what-you-get">
        <div className="container">
          <h2 id="about-what-you-get" className={styles.featuresHeading}>
            What you get
          </h2>
        </div>

        {prefersReducedMotion ? <FeatureMarqueeStatic /> : <FeatureMarqueeAnimated />}
      </section>
    </>
  );
}
