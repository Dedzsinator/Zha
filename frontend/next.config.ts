import type { NextConfig } from "next";

const backendUrl = process.env.BACKEND_URL ?? 'http://localhost:8000';

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: '/generate/:path*',
        destination: `${backendUrl}/generate/:path*`,
      },
      {
        source: '/download/:path*',
        destination: `${backendUrl}/download/:path*`,
      },
    ];
  },
};

export default nextConfig;
