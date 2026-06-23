import React from "react";
import { Link, NavLink } from "react-router-dom";
import { Navbar, Container, Nav, Button, Dropdown } from "react-bootstrap";
import { FaRegMoon, FaRegSun } from "react-icons/fa6";
import { useTheme } from "../layout/theme";
import { useTranslation } from "react-i18next";
import { AUTONYM_MAP, type LanguageCode, isRtlLanguage } from "../i18n/languages";
import { getLanguageOptionsByUsage } from "../i18n/translation-priority";
import { useAuth } from "../auth/context";

/**
 * Top navigation: brand, primary routes (chat at `/`, `/mission`), an
 * auth-aware cluster (Account/Sign out when signed in, else Sign in/up),
 * the theme toggle, and the language selector.
 *
 * The language picker is ordered by **estimated usage**
 * (getLanguageOptionsByUsage), not alphabetically — a deliberate choice that
 * foregrounds helexa's international grounding. Each item shows the autonym
 * (language in its own script) plus a secondary label in the current
 * language; RTL-aware alignment.
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
      className="app-header border-bottom"
      variant={theme === "dark" ? "dark" : "light"}
    >
      <Container fluid>
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

          <div className="d-flex align-items-center gap-2">
            {/* Auth-aware cluster. */}
            {status === "authed" ? (
              <>
                <NavLink to="/account" className="nav-link">
                  {t("nav.account")}
                </NavLink>
                <Button
                  size="sm"
                  variant="outline-secondary"
                  onClick={logout}
                  className="me-1"
                >
                  {t("nav.logout")}
                </Button>
              </>
            ) : (
              <>
                <NavLink to="/login" className="nav-link">
                  {t("nav.login")}
                </NavLink>
                <NavLink to="/register" className="nav-link">
                  {t("nav.register")}
                </NavLink>
              </>
            )}

            <Button
              size="sm"
              variant="outline-secondary"
              type="button"
              onClick={toggleTheme}
              aria-label={
                theme === "dark"
                  ? t("theme.toggle.toLight")
                  : t("theme.toggle.toDark")
              }
              className="d-inline-flex align-items-center justify-content-center"
            >
              {theme === "dark" ? <FaRegSun size={16} /> : <FaRegMoon size={16} />}
            </Button>

            <Dropdown
              align={isRtl ? "start" : "end"}
              className={theme === "dark" ? "dropdown-menu-dark-context" : ""}
            >
              <Dropdown.Toggle
                size="sm"
                variant={theme === "dark" ? "secondary" : "outline-secondary"}
                id="language-switcher"
              >
                <span className="me-1" aria-hidden="true">
                  文A
                </span>
                <span>{AUTONYM_MAP[currentLanguage]}</span>
              </Dropdown.Toggle>
              <Dropdown.Menu
                className={theme === "dark" ? "dropdown-menu-dark" : ""}
              >
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
