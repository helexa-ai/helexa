import { Navigate, useLocation } from "react-router-dom";

/**
 * `/register` — kept as a permanent redirect to the unified `/auth` page,
 * landing on the sign-up tab. Preserves any other query params.
 */
export default function Register() {
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  params.set("tab", "signup");
  return <Navigate to={`/auth?${params.toString()}`} replace />;
}
