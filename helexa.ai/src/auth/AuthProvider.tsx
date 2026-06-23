import { useEffect, useState, type ReactNode } from "react";
import { accountApi } from "../api/account";
import { claimAnonymousData } from "../data/repositories";
import { getFingerprint } from "../lib/fingerprint";
import { AuthContext } from "./context";

const TOKEN_KEY = "helexa.token";
const EMAIL_KEY = "helexa.email";

export default function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem(TOKEN_KEY),
  );
  const [email, setEmail] = useState<string | null>(() =>
    localStorage.getItem(EMAIL_KEY),
  );
  const [accountId, setAccountId] = useState<string | null>(null);

  // Resolve the account id for an existing session (page reload) so the chat
  // workspace can scope its IndexedDB owner without a fresh login.
  useEffect(() => {
    if (!token || accountId) return;
    accountApi()
      .account(token)
      .then((a) => setAccountId(a.account_id))
      .catch(() => {
        /* token may be stale; chat falls back to anon until re-login */
      });
  }, [token, accountId]);

  async function login(em: string, password: string): Promise<void> {
    const api = accountApi();
    const session = await api.login(em, password);
    localStorage.setItem(TOKEN_KEY, session.token);
    localStorage.setItem(EMAIL_KEY, em);
    setToken(session.token);
    setEmail(em);
    // Claim anonymous local history into the account (stays client-side).
    try {
      const acct = await api.account(session.token);
      setAccountId(acct.account_id);
      await claimAnonymousData(acct.account_id);
    } catch {
      /* non-fatal */
    }
  }

  async function register(em: string, password: string): Promise<void> {
    const fingerprint = await getFingerprint();
    await accountApi().register(em, password, fingerprint);
  }

  function logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EMAIL_KEY);
    setToken(null);
    setEmail(null);
    setAccountId(null);
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        email,
        accountId,
        status: token ? "authed" : "anon",
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
