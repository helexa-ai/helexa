// Which audience is looking at this deployment.
//
// The same build is served on the public domain and on the internal
// mesh, so "is this the internal site?" is a question about the address
// in the address bar, not about the bundle. Nothing here gates access —
// every route works on every host — it only decides what gets
// advertised in the interface.

/**
 * True when the page is being viewed over the internal mesh or a local
 * development server.
 *
 * Deliberately not a security boundary. Anyone can set a hosts entry, and
 * the routes this hides are reachable by typing the URL on any host.
 * Use it for "not ready to promote yet", never for "not allowed to see".
 */
export function isInternalHost(): boolean {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  return (
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "[::1]" ||
    host.endsWith(".internal")
  );
}
