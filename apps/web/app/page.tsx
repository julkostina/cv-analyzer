"use client";

import { AboutSection } from "../components/about-section/AboutSection";
import { HomeAnalyzer } from "../components/home/HomeAnalyzer";
import { PageLayout } from "../components/page-layout/PageLayout";

export default function HomePage() {
  return (
    <PageLayout>
      <AboutSection />
      <HomeAnalyzer />
    </PageLayout>
  );
}
