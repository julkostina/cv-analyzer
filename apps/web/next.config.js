/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    // Old URL when history lived under app/pages/; literal `app/pages` breaks Next dev (Pages Router collision).
    return [{ source: "/pages/history", destination: "/history", permanent: true }];
  },
};

export default nextConfig;
