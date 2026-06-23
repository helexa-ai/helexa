import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ThemeContext, type ThemeMode, type ThemeContextValue } from "./theme";

const THEME_STORAGE_KEY = "helexa.theme";

/**
 * Detect initial theme:
 * 1. Use explicit user preference from localStorage, if present
 * 2. Otherwise, respect system prefers-color-scheme
 * 3. Fallback to dark if nothing else is available
 */
function detectInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "dark";
  }

  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "light" || stored === "dark") {
    return stored;
  }

  const prefersDark = window.matchMedia?.(
    "(prefers-color-scheme: dark)",
  ).matches;

  return prefersDark ? "dark" : "light";
}

export interface ThemeProviderProps {
  children: ReactNode;
}

/**
 * ThemeProvider
 *
 * - Exposes theme + controls via ThemeContext
 * - Syncs theme to <html data-theme="..."> for CSS
 * - Persists explicit user choice in localStorage
 * - Listens to system theme changes when there is no explicit user override
 *
 * NOTE: This file exports only a single React component to keep
 * fast-refresh compatibility with the React tooling.
 */
const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeMode>(() =>
    detectInitialTheme(),
  );

  // Keep <html> data attribute in sync so CSS can style via [data-theme]
  useEffect(() => {
    if (typeof document === "undefined") return;

    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  // React to system theme changes if there is no explicit user override
  useEffect(() => {
    if (typeof window === "undefined") return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (event: MediaQueryListEvent) => {
      const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
      // Only auto-update if user hasn't explicitly chosen a theme
      if (!stored) {
        setThemeState(event.matches ? "dark" : "light");
      }
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handleChange);
    } else {
      mediaQuery.addListener(handleChange);
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener("change", handleChange);
      } else {
        mediaQuery.removeListener(handleChange);
      }
    };
  }, []);

  const setTheme = useCallback<ThemeContextValue["setTheme"]>(
    (nextThemeOrUpdater) => {
      setThemeState((prev) => {
        const value =
          typeof nextThemeOrUpdater === "function"
            ? (nextThemeOrUpdater as (p: ThemeMode) => ThemeMode)(prev)
            : nextThemeOrUpdater;
        window.localStorage.setItem(THEME_STORAGE_KEY, value);
        return value;
      });
    },
    [],
  );

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, [setTheme]);

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme,
      setTheme,
      toggleTheme,
    }),
    [theme, setTheme, toggleTheme],
  );

  return (
    <ThemeContext.Provider value={value}>
      <div className={`app-root theme-${theme}`}>{children}</div>
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;
