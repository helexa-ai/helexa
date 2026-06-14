/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base origin of the bench API. Empty → use the dev proxy / same origin. */
  readonly VITE_API_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
