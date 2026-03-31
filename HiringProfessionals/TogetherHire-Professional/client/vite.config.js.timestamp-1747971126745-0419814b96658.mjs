// vite.config.js
import react from "file:///home/ubuntu/hiring-game/client/node_modules/@vitejs/plugin-react/dist/index.mjs";
import builtins from "file:///home/ubuntu/hiring-game/client/node_modules/rollup-plugin-polyfill-node/dist/index.js";
import { defineConfig, searchForWorkspaceRoot } from "file:///home/ubuntu/hiring-game/client/node_modules/vite/dist/node/index.js";
import restart from "file:///home/ubuntu/hiring-game/client/node_modules/vite-plugin-restart/dist/index.js";
import UnoCSS from "file:///home/ubuntu/hiring-game/client/node_modules/unocss/dist/vite.mjs";
import dns from "dns";
dns.setDefaultResultOrder("verbatim");
var builtinsPlugin = {
  ...builtins({ include: ["fs/promises"] }),
  name: "rollup-plugin-polyfill-node"
};
var vite_config_default = defineConfig({
  optimizeDeps: {
    exclude: ["@empirica/tajriba", "@empirica/core"]
  },
  server: {
    port: 8844,
    open: false,
    strictPort: true,
    host: "0.0.0.0",
    hmr: {
      host: "localhost",
      protocol: "ws",
      port: 8844
    },
    fs: {
      allow: [
        // search up for workspace root
        searchForWorkspaceRoot(process.cwd())
      ]
    }
  },
  build: {
    minify: false,
    target: "esnext",
    sourcemap: true,
    rollupOptions: {
      preserveEntrySignatures: "strict",
      plugins: [builtinsPlugin],
      output: {
        sourcemap: true
      }
    }
  },
  clearScreen: false,
  plugins: [
    restart({
      restart: [
        "./uno.config.cjs",
        "./node_modules/@empirica/core/dist/**/*.{js,ts,jsx,tsx,css}",
        "./node_modules/@empirica/core/assets/**/*.css"
      ]
    }),
    UnoCSS(),
    react()
  ],
  define: {
    "process.env": {
      NODE_ENV: process.env.NODE_ENV || "development"
    }
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCIvaG9tZS91YnVudHUvaGlyaW5nLWdhbWUvY2xpZW50XCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCIvaG9tZS91YnVudHUvaGlyaW5nLWdhbWUvY2xpZW50L3ZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9ob21lL3VidW50dS9oaXJpbmctZ2FtZS9jbGllbnQvdml0ZS5jb25maWcuanNcIjtpbXBvcnQgcmVhY3QgZnJvbSBcIkB2aXRlanMvcGx1Z2luLXJlYWN0XCI7XG5pbXBvcnQgYnVpbHRpbnMgZnJvbSBcInJvbGx1cC1wbHVnaW4tcG9seWZpbGwtbm9kZVwiO1xuaW1wb3J0IHsgZGVmaW5lQ29uZmlnLCBzZWFyY2hGb3JXb3Jrc3BhY2VSb290IH0gZnJvbSBcInZpdGVcIjtcbmltcG9ydCByZXN0YXJ0IGZyb20gXCJ2aXRlLXBsdWdpbi1yZXN0YXJ0XCI7XG5pbXBvcnQgVW5vQ1NTIGZyb20gXCJ1bm9jc3Mvdml0ZVwiO1xuaW1wb3J0IGRucyBmcm9tIFwiZG5zXCI7XG5cbmRucy5zZXREZWZhdWx0UmVzdWx0T3JkZXIoXCJ2ZXJiYXRpbVwiKTtcblxuY29uc3QgYnVpbHRpbnNQbHVnaW4gPSB7XG4gIC4uLmJ1aWx0aW5zKHsgaW5jbHVkZTogW1wiZnMvcHJvbWlzZXNcIl0gfSksXG4gIG5hbWU6IFwicm9sbHVwLXBsdWdpbi1wb2x5ZmlsbC1ub2RlXCIsXG59O1xuXG4vLyBodHRwczovL3ZpdGVqcy5kZXYvY29uZmlnL1xuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKHtcbiAgb3B0aW1pemVEZXBzOiB7XG4gICAgZXhjbHVkZTogW1wiQGVtcGlyaWNhL3RhanJpYmFcIiwgXCJAZW1waXJpY2EvY29yZVwiXSxcbiAgfSxcbiAgc2VydmVyOiB7XG4gICAgcG9ydDogODg0NCxcbiAgICBvcGVuOiBmYWxzZSxcbiAgICBzdHJpY3RQb3J0OiB0cnVlLFxuICAgIGhvc3Q6IFwiMC4wLjAuMFwiLFxuICAgIGhtcjoge1xuICAgICAgaG9zdDogXCJsb2NhbGhvc3RcIixcbiAgICAgIHByb3RvY29sOiBcIndzXCIsXG4gICAgICBwb3J0OiA4ODQ0LFxuICAgIH0sXG4gICAgZnM6IHtcbiAgICAgIGFsbG93OiBbXG4gICAgICAgIC8vIHNlYXJjaCB1cCBmb3Igd29ya3NwYWNlIHJvb3RcbiAgICAgICAgc2VhcmNoRm9yV29ya3NwYWNlUm9vdChwcm9jZXNzLmN3ZCgpKSxcbiAgICAgIF0sXG4gICAgfSxcbiAgfSxcbiAgYnVpbGQ6IHtcbiAgICBtaW5pZnk6IGZhbHNlLFxuICAgIHRhcmdldDogXCJlc25leHRcIixcbiAgICBzb3VyY2VtYXA6IHRydWUsXG4gICAgcm9sbHVwT3B0aW9uczoge1xuICAgICAgcHJlc2VydmVFbnRyeVNpZ25hdHVyZXM6IFwic3RyaWN0XCIsXG4gICAgICBwbHVnaW5zOiBbYnVpbHRpbnNQbHVnaW5dLFxuICAgICAgb3V0cHV0OiB7XG4gICAgICAgIHNvdXJjZW1hcDogdHJ1ZSxcbiAgICAgIH0sXG4gICAgfSxcbiAgfSxcbiAgY2xlYXJTY3JlZW46IGZhbHNlLFxuICBwbHVnaW5zOiBbXG4gICAgcmVzdGFydCh7XG4gICAgICByZXN0YXJ0OiBbXG4gICAgICAgIFwiLi91bm8uY29uZmlnLmNqc1wiLFxuICAgICAgICBcIi4vbm9kZV9tb2R1bGVzL0BlbXBpcmljYS9jb3JlL2Rpc3QvKiovKi57anMsdHMsanN4LHRzeCxjc3N9XCIsXG4gICAgICAgIFwiLi9ub2RlX21vZHVsZXMvQGVtcGlyaWNhL2NvcmUvYXNzZXRzLyoqLyouY3NzXCIsXG4gICAgICBdLFxuICAgIH0pLFxuICAgIFVub0NTUygpLFxuICAgIHJlYWN0KCksXG4gIF0sXG4gIGRlZmluZToge1xuICAgIFwicHJvY2Vzcy5lbnZcIjoge1xuICAgICAgTk9ERV9FTlY6IHByb2Nlc3MuZW52Lk5PREVfRU5WIHx8IFwiZGV2ZWxvcG1lbnRcIixcbiAgICB9LFxuICB9LFxufSk7XG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQStRLE9BQU8sV0FBVztBQUNqUyxPQUFPLGNBQWM7QUFDckIsU0FBUyxjQUFjLDhCQUE4QjtBQUNyRCxPQUFPLGFBQWE7QUFDcEIsT0FBTyxZQUFZO0FBQ25CLE9BQU8sU0FBUztBQUVoQixJQUFJLHNCQUFzQixVQUFVO0FBRXBDLElBQU0saUJBQWlCO0FBQUEsRUFDckIsR0FBRyxTQUFTLEVBQUUsU0FBUyxDQUFDLGFBQWEsRUFBRSxDQUFDO0FBQUEsRUFDeEMsTUFBTTtBQUNSO0FBR0EsSUFBTyxzQkFBUSxhQUFhO0FBQUEsRUFDMUIsY0FBYztBQUFBLElBQ1osU0FBUyxDQUFDLHFCQUFxQixnQkFBZ0I7QUFBQSxFQUNqRDtBQUFBLEVBQ0EsUUFBUTtBQUFBLElBQ04sTUFBTTtBQUFBLElBQ04sTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLElBQ1osTUFBTTtBQUFBLElBQ04sS0FBSztBQUFBLE1BQ0gsTUFBTTtBQUFBLE1BQ04sVUFBVTtBQUFBLE1BQ1YsTUFBTTtBQUFBLElBQ1I7QUFBQSxJQUNBLElBQUk7QUFBQSxNQUNGLE9BQU87QUFBQTtBQUFBLFFBRUwsdUJBQXVCLFFBQVEsSUFBSSxDQUFDO0FBQUEsTUFDdEM7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0EsT0FBTztBQUFBLElBQ0wsUUFBUTtBQUFBLElBQ1IsUUFBUTtBQUFBLElBQ1IsV0FBVztBQUFBLElBQ1gsZUFBZTtBQUFBLE1BQ2IseUJBQXlCO0FBQUEsTUFDekIsU0FBUyxDQUFDLGNBQWM7QUFBQSxNQUN4QixRQUFRO0FBQUEsUUFDTixXQUFXO0FBQUEsTUFDYjtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFDQSxhQUFhO0FBQUEsRUFDYixTQUFTO0FBQUEsSUFDUCxRQUFRO0FBQUEsTUFDTixTQUFTO0FBQUEsUUFDUDtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsTUFDRjtBQUFBLElBQ0YsQ0FBQztBQUFBLElBQ0QsT0FBTztBQUFBLElBQ1AsTUFBTTtBQUFBLEVBQ1I7QUFBQSxFQUNBLFFBQVE7QUFBQSxJQUNOLGVBQWU7QUFBQSxNQUNiLFVBQVUsUUFBUSxJQUFJLFlBQVk7QUFBQSxJQUNwQztBQUFBLEVBQ0Y7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
