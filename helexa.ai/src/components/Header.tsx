import React from "react";
import { Link, NavLink } from "react-router-dom";
import { Navbar, Container, Nav, Dropdown } from "react-bootstrap";
import { FaRegMoon, FaRegSun, FaGithub, FaRegUser, FaCircleUser } from "react-icons/fa6";
import { useTheme } from "../layout/theme";
import { useTranslation } from "react-i18next";
import { AUTONYM_MAP, type LanguageCode, isRtlLanguage } from "../i18n/languages";
import { getLanguageOptionsByUsage } from "../i18n/translation-priority";
import { useAuth } from "../auth/context";
import { accountApi } from "../api/account";
import { isInternalHost } from "../lib/host";

/**
 * Top navigation: brand, primary routes (chat at `/`, `/mission`), an
 * auth-aware cluster (Account/Sign out when signed in, else Sign in +
 * a Sign-up pill), then a quiet icon cluster: GitHub, theme toggle,
 * language selector. Icon buttons are borderless (`hx-icon-btn`) so the
 * header stays calm; the sign-up pill is the single emphasised control.
 *
 * The language picker is ordered by **estimated usage**
 * (getLanguageOptionsByUsage), not alphabetically — a deliberate choice
 * that foregrounds helexa's international grounding. Each item shows the
 * autonym plus a secondary label in the current language; RTL-aware.
 */
const Header: React.FC = () => {
  const { theme, toggleTheme } = useTheme();
  const { t, i18n } = useTranslation("common");
  const { status, token, logout } = useAuth();

  // Whether to offer the investor portal. The server answers with a
  // boolean and nothing else (see AccountBalance.angel_access): this
  // bundle is public, so it must never learn round names.
  //
  // Keyed on the token rather than a bare boolean: signing out and back in
  // as somebody else would otherwise show the previous account's answer for
  // a render. Storing the token alongside the result means a stale value
  // simply doesn't match and is ignored — and it removes the synchronous
  // reset that made this a setState-in-effect.
  const [angel, setAngel] = React.useState<{ token: string; access: boolean } | null>(
    null,
  );
  React.useEffect(() => {
    if (status !== "authed" || !token) return;
    let cancelled = false;
    accountApi()
      .account(token)
      .then((a) => {
        if (!cancelled) setAngel({ token, access: a.angel_access === true });
      })
      // Silent: a missing link is a small inconvenience for someone who
      // still holds the invitation link they were sent, whereas an error
      // banner in the header would be a puzzle for everyone else.
      .catch(() => {});
    return (): void => {
      cancelled = true;
    };
  }, [status, token]);
  const angelAccess = angel?.token === token && angel.access;

  const currentLanguage: LanguageCode = (i18n.language.split("-")[0] ||
    "en") as LanguageCode;
  const isRtl = isRtlLanguage(currentLanguage);
  const languageOptions = getLanguageOptionsByUsage();

  return (
    <Navbar
      expand="lg"
      className="app-header"
      variant={theme === "dark" ? "dark" : "light"}
    >
      <Container fluid className="px-4">
        <Navbar.Brand
          as={Link}
          to="/"
          className="d-flex align-items-center gap-2"
        >
          <img
            src="/logo.png"
            alt="helexa logo"
            width={28}
            height={28}
            style={{ borderRadius: "999px" }}
          />
          <span className="fw-semibold text-uppercase small tracking-wide">
            {t("app.name")}
          </span>
        </Navbar.Brand>

        <Navbar.Toggle aria-controls="main-navbar" />

        <Navbar.Collapse id="main-navbar">
          <Nav className="me-auto">
            <NavLink
              to="/chat"
              className={({ isActive }): string =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {t("nav.chat")}
            </NavLink>
            <NavLink
              to="/images"
              className={({ isActive }): string =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {t("nav.images")}
            </NavLink>
            {/* Documentation is reachable on every host — this only
                decides whether it is advertised. The pages are still
                being written and proofread, so the link is shown on the
                internal mesh and in development, and anyone who knows
                the URL can still open /docs on the public site. Remove
                the condition once the content is ready to promote. */}
            {isInternalHost() && (
              <NavLink
                to="/docs"
                className={({ isActive }): string =>
                  isActive ? "nav-link active" : "nav-link"
                }
              >
                {t("nav.docs")}
              </NavLink>
            )}
            <NavLink
              to="/mission"
              className={({ isActive }): string =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {t("nav.mission")}
            </NavLink>
          </Nav>

          <div className="d-flex align-items-center gap-1">
            {/* One user control for the whole auth surface.
                Anonymous: the icon is a plain link straight to /auth,
                where sign in and sign up are tabs — one click instead of
                a menu whose two items were themselves links.
                Signed in: a small menu (account / sign out). Its items are
                plain <Link>/<button> with the dropdown-item class rather
                than `Dropdown.Item as={Link}` — that indirection rendered
                inert anchors (no href, so no pointer cursor and no
                navigation), which is what broke sign in/up in the first
                place. */}
            {status === "authed" ? (
              <Dropdown align={isRtl ? "start" : "end"}>
                <Dropdown.Toggle
                  as="button"
                  type="button"
                  className="hx-icon-btn hx-user-authed"
                  id="user-menu"
                  aria-label={t("nav.account")}
                >
                  <FaCircleUser size={18} />
                </Dropdown.Toggle>
                <Dropdown.Menu>
                  <Link className="dropdown-item" to="/account">
                    {t("nav.account")}
                  </Link>
                  {angelAccess && (
                    /* Label is the hostname itself — language-neutral, so
                       it needs no i18n key and cannot leave 41 locales
                       failing `npm run i18n:check`. Same reasoning as the
                       numeric eyebrows in Mission.tsx. It is a separate
                       origin with its own session, so a plain anchor. */
                    <a
                      className="dropdown-item"
                      href="https://angels.helexa.ai"
                      rel="noreferrer"
                    >
                      angels.helexa.ai
                    </a>
                  )}
                  <button
                    type="button"
                    className="dropdown-item"
                    onClick={logout}
                  >
                    {t("nav.logout")}
                  </button>
                </Dropdown.Menu>
              </Dropdown>
            ) : (
              <>
                {/* Signing up was an unlabelled glyph — the primary
                    conversion action with no words on it. `nav.register`
                    already exists in every shipped locale, so labelling it
                    costs no translation. */}
                <Link
                  to="/auth"
                  className="hx-icon-btn"
                  id="user-menu"
                  aria-label={t("nav.login")}
                  title={t("nav.login")}
                >
                  <FaRegUser size={16} />
                </Link>
                <Link to="/auth?tab=signup" className="hx-header-signup">
                  {t("nav.register")}
                </Link>
              </>
            )}

            <a
              href="https://github.com/helexa-ai"
              target="_blank"
              rel="noreferrer"
              className="hx-icon-btn"
              aria-label="GitHub"
            >
              <FaGithub size={17} />
            </a>

            <button
              type="button"
              className="hx-icon-btn"
              onClick={toggleTheme}
              aria-label={
                theme === "dark"
                  ? t("theme.toggle.toLight")
                  : t("theme.toggle.toDark")
              }
            >
              {theme === "dark" ? <FaRegSun size={16} /> : <FaRegMoon size={16} />}
            </button>

            <Dropdown align={isRtl ? "start" : "end"}>
              <Dropdown.Toggle
                as="button"
                type="button"
                className="hx-icon-btn hx-icon-btn-wide"
                id="language-switcher"
              >
                <span aria-hidden="true">文A</span>
                <span>{AUTONYM_MAP[currentLanguage]}</span>
              </Dropdown.Toggle>
              <Dropdown.Menu>
                {languageOptions.map(({ code, autonym }) => (
                  <Dropdown.Item
                    key={code}
                    active={code === currentLanguage}
                    onClick={() => void i18n.changeLanguage(code)}
                    className="d-flex align-items-center gap-2"
                  >
                    <span>{autonym}</span>
                    <span className="text-muted small fw-light">
                      · {t(`lang.${code}`)}
                    </span>
                  </Dropdown.Item>
                ))}
              </Dropdown.Menu>
            </Dropdown>
          </div>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default Header;
