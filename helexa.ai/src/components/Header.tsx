import React from "react";
import { Link, NavLink } from "react-router-dom";
import { Navbar, Container, Nav, Dropdown } from "react-bootstrap";
import { FaRegMoon, FaRegSun, FaGithub } from "react-icons/fa6";
import { useTheme } from "../layout/theme";
import { useTranslation } from "react-i18next";
import { AUTONYM_MAP, type LanguageCode, isRtlLanguage } from "../i18n/languages";
import { getLanguageOptionsByUsage } from "../i18n/translation-priority";
import { useAuth } from "../auth/context";

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
  const { status, logout } = useAuth();

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
              to="/"
              end
              className={({ isActive }): string =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {t("nav.chat")}
            </NavLink>
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
            {/* Auth-aware cluster. */}
            {status === "authed" ? (
              <>
                <NavLink to="/account" className="nav-link">
                  {t("nav.account")}
                </NavLink>
                <button
                  type="button"
                  className="hx-icon-btn hx-icon-btn-wide"
                  onClick={logout}
                >
                  {t("nav.logout")}
                </button>
              </>
            ) : (
              <>
                <NavLink to="/login" className="nav-link">
                  {t("nav.login")}
                </NavLink>
                <NavLink to="/register" className="hx-pill-cta mx-1">
                  {t("nav.register")}
                </NavLink>
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
