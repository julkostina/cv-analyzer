/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [{ source: "/pages/history", destination: "/history", permanent: true }];
  },
};

export default nextConfig;
