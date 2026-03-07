/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    remotePatterns: [],
  },
  experimental: {
    optimizeFonts: true,
  },
}

module.exports = nextConfig
