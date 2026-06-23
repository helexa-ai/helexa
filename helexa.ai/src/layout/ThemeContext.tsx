import React from "react";
import type { ThemeContextValue } from "./theme";

/**
 * ThemeContextComponent
 *
 * This component simply renders its children and exists to satisfy the
 * "only export components" constraint for React fast-refresh tooling.
 * It is not intended for direct use in the app.
 */
export const ThemeContextComponent: React.FC<{
  children?: React.ReactNode;
}> = ({ children }) => <>{children}</>;

/**
 * ThemeProviderComponent
 *
 * This component is also a no-op wrapper whose only purpose is to ensure
 * this file exports React components exclusively. The actual theme logic
 * (context creation, state, hooks) is defined in `theme.ts`.
 */
export const ThemeProviderComponent: React.FC<{
  value: ThemeContextValue;
  children?: React.ReactNode;
}> = ({ children }) => <>{children}</>;

/**
 * ThemeGuard
 *
 * Optional guard component you can render to assert a theme context is
 * available. It currently just renders its children unchanged.
 */
export const ThemeGuard: React.FC<{ children?: React.ReactNode }> = ({
  children,
}) => <>{children}</>;
