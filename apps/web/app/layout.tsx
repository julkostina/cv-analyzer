import type { Metadata } from "next";
import { Noto_Serif_Display, Roboto } from "next/font/google";
import { SiteShell } from "../components/SiteShell";
import "./globals.css";

const roboto = Roboto({
  subsets: ["latin", "cyrillic"],
  weight: ["400", "500", "700"],
  variable: "--font-body",
  display: "swap",
});

const notoSerifDisplay = Noto_Serif_Display({
  subsets: ["latin", "cyrillic"],
  weight: ["400", "600", "700", "900"],
  variable: "--font-title",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Аналізатор резюме",
  description: "Оцінка відповідності резюме вакансії та рекомендації",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="uk" className={`${roboto.variable} ${notoSerifDisplay.variable}`}>
      <body>
        <SiteShell>{children}</SiteShell>
      </body>
    </html>
  );
}
