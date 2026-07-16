import React from "react";
import { Container } from "react-bootstrap";

/**
 * AuthCard
 *
 * Shared shell for the auth flows (login, register, verify, reset):
 * a narrow centered surface with the helix mark and a title, so every
 * auth route carries the same visual grammar as the rest of the site.
 */
const AuthCard: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <Container className="py-5 flex-grow-1">
    <div className="hx-auth-card">
      <img src="/logo.png" alt="" aria-hidden="true" className="hx-auth-logo" />
      <h1 className="h4 mb-4">{title}</h1>
      {children}
    </div>
  </Container>
);

export default AuthCard;
