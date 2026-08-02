import { useSearchParams } from "react-router-dom";
import { Nav } from "react-bootstrap";
import { useTranslation } from "react-i18next";
import AuthCard from "../../components/AuthCard";
import { SignInForm, SignUpForm } from "./forms";

/**
 * `/auth` — the single entry point to the account surface.
 *
 * Sign in and sign up sit behind two tabs on one route rather than two
 * routes joined by easy-to-miss cross-links: arriving here the visitor
 * sees both options at once and signing in — the common case for anyone
 * who already has an account — is the default.
 *
 * The active tab lives in the query string (`?tab=signup`) so the choice
 * is linkable, bookmarkable and survives the back button; `?next=` is
 * preserved by the router, so RequireAuth's redirect still returns the
 * visitor where they were headed.
 *
 * Tab labels reuse the existing `login.title` / `register.submit`
 * strings, so the page is fully translated in every shipped locale
 * without adding keys that would sit untranslated in 41 of them.
 */
export default function Auth() {
  const { t } = useTranslation("account");
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "signup" ? "signup" : "signin";

  function select(next: string | null) {
    if (!next || next === tab) return;
    const updated = new URLSearchParams(params);
    if (next === "signup") {
      updated.set("tab", "signup");
    } else {
      updated.delete("tab");
    }
    // replace: switching tabs is a change of view, not a new place —
    // Back should leave the auth page, not walk the tabs.
    setParams(updated, { replace: true });
  }

  return (
    <AuthCard
      title={tab === "signup" ? t("register.title") : t("login.title")}
    >
      <Nav
        variant="tabs"
        activeKey={tab}
        onSelect={select}
        className="hx-auth-tabs mb-4"
      >
        <Nav.Item>
          <Nav.Link eventKey="signin">{t("login.title")}</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="signup">{t("register.submit")}</Nav.Link>
        </Nav.Item>
      </Nav>

      {tab === "signup" ? <SignUpForm /> : <SignInForm />}
    </AuthCard>
  );
}
