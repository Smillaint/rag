import { defineConfig } from "vitest/config";
import { cloudflareTest } from "@cloudflare/vitest-plugin";

process.env.CLOUDFLARE_LOAD_DEV_VARS_FROM_DOT_ENV = "false";

export default defineConfig({
  test: {
    setupFiles: ["./test/setup.js"],
  },
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: {
          RAG_GATEWAY_API_KEY: "test-gateway-key",
          RAG_ORIGIN_API_KEY: "test-origin-key",
        },
      },
    }),
  ],
});
