import type { Metadata } from "next";
import { Noto_Serif_Display, Roboto } from "next/font/google";
import { SiteShell } from "../components/SiteShell";
import "./globals.css";

const roboto = Roboto({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-body",
  display: "swap",
});

const notoSerifDisplay = Noto_Serif_Display({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-title",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CV Analyzer",
  description: "Resume–job match scoring and recommendations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${roboto.variable} ${notoSerifDisplay.variable}`}>
      <body>
        <SiteShell>{children}</SiteShell>
      </body>
    </html>
  );
}
