import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import type { ReactNode } from "react";
import "./globals.css";

const serif = localFont({
  src: [
    { path: "../fonts/fraunces-normal-latin.woff2", weight: "300 700", style: "normal" },
    { path: "../fonts/fraunces-italic-latin.woff2", weight: "300 700", style: "italic" },
  ],
  variable: "--font-serif",
  display: "swap",
  fallback: ["Iowan Old Style", "Palatino Linotype", "Georgia", "serif"],
});

const sans = localFont({
  src: [
    { path: "../fonts/instrument-sans-normal-latin.woff2", weight: "400 700", style: "normal" },
    { path: "../fonts/instrument-sans-italic-latin.woff2", weight: "400 700", style: "italic" },
  ],
  variable: "--font-sans",
  display: "swap",
  fallback: ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
});

export const metadata: Metadata = {
  title: "Research Observatory",
  description: "A private research workspace by Arjunworks: ask questions of public papers and inspect the cited pages.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#0b1a1f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable}`}>
      <body>{children}</body>
    </html>
  );
}
