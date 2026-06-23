import { createContext, useContext } from "react";

export interface AuthContextValue {
  token: string | null;
  email: string | null;
  /** The signed-in account id (for the Dexie owner + usage queries). */
  accountId: string | null;
  status: "anon" | "authed";
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue>({
  token: null,
  email: null,
  accountId: null,
  status: "anon",
  login: async () => {},
  register: async () => {},
  logout: () => {},
});

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}
