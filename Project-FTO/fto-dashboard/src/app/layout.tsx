import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { SessionProvider } from "next-auth/react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FTO Dashboard - PMI",
  description: "Factory Training Operations Dashboard - Performance Management Interface",
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body className={`${inter.className} bg-slate-950 text-slate-100`}>
        <SessionProvider>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
