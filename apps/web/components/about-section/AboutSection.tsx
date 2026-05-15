"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useLayoutEffect, useRef, useState } from "react";
import { uk } from "../../lib/strings-uk";
import styles from "./AboutSection.module.css";

const FEATURES = uk.about.features;

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
    <div className={styles.carouselViewport} role="region" aria-label={uk.about.carouselAria}>
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
  const a = uk.about;

  return (
    <>
      <section className={styles.intro} aria-labelledby="about-heading">
        <div className={`container ${styles.introContent}`}>
          <h1 id="about-heading" className={styles.introTitle}>
            {a.heading}
          </h1>
          <p className={styles.introLead}>{a.lead1}</p>
          <p className={styles.introLead}>{a.lead2}</p>
          <p className={styles.introNote}>{a.note}</p>
        </div>
      </section>

      <section className={styles.featuresSection} aria-labelledby="about-what-you-get">
        <div className="container">
          <h2 id="about-what-you-get" className={styles.featuresHeading}>
            {a.featuresHeading}
          </h2>
        </div>

        {prefersReducedMotion ? <FeatureMarqueeStatic /> : <FeatureMarqueeAnimated />}
      </section>
    </>
  );
}
