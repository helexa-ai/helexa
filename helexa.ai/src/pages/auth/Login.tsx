import { Navigate, useLocation } from "react-router-dom";

/**
 * `/login` — kept as a permanent redirect to the unified `/auth` page.
 *
 * Old links live on: bookmarks, RequireAuth's `?next=` redirect, and the
 * "back to sign in" links in the verify/reset flows. The query string is
 * carried across so `?next=` still lands the visitor where they meant to go.
 */
export default function Login() {
  const { search } = useLocation();
  return <Navigate to={`/auth${search}`} replace />;
}
