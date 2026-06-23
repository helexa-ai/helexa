/* helexa.ai/src/layout/theme.ts
 *
 * Shared theme types, context value shape, and convenience hooks.
 *
 * This module intentionally does NOT export any React components.
 * Keeping it free of components means it can be safely imported by
 * other modules without violating the "only export components" rule
 * used by the React fast-refresh tooling.
 */

import { createContext, useContext } from "react";

/**
 * ThemeMode
 *
 * The high-level mode of the UI. Extend this if you add more variants
 * (e.g. "system", "high-contrast", etc.).
 */
export type ThemeMode = "light" | "dark";

/**
 * ThemeContextValue
 *
 * The shape of the theme context shared across the app.
 * - `theme`: current theme mode
 * - `setTheme`: update theme (accepts value or updater fn)
 * - `toggleTheme`: quick helper to switch between modes
 */
export interface ThemeContextValue {
  theme: ThemeMode;
  setTheme: (theme: ThemeMode | ((prev: ThemeMode) => ThemeMode)) => void;
  toggleTheme: () => void;
}

/**
 * ThemeContext
 *
 * Default placeholder values are no-ops. A real provider should wrap the app
 * and supply the actual stateful implementation.
 */
export const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark",
  setTheme: () => {},
  toggleTheme: () => {},
});

/**
 * useTheme
 *
 * Convenience hook to access the theme context.
 */
export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}

/**
 * useThemeMode
 *
 * Convenience hook for just the current theme mode.
 */
export function useThemeMode(): ThemeMode {
  return useContext(ThemeContext).theme;
}

/**
 * useToggleTheme
 *
 * Convenience hook that returns only the toggle function.
 */
export function useToggleTheme(): () => void {
  return useContext(ThemeContext).toggleTheme;
}
