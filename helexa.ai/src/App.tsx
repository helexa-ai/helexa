import { BrowserRouter, Routes, Route } from "react-router-dom";
import ThemeProvider from "./layout/ThemeProvider";
import AuthProvider from "./auth/AuthProvider";
import RequireAuth from "./auth/RequireAuth";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Mission from "./pages/Mission";
import Chat from "./pages/Chat";
import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import VerifyEmail from "./pages/auth/VerifyEmail";
import RequestReset from "./pages/auth/RequestReset";
import ResetPassword from "./pages/auth/ResetPassword";
import Dashboard from "./pages/account/Dashboard";
import ApiKeys from "./pages/account/ApiKeys";
import "./App.css";

// Composition root: theme → router → auth → layout shell. `/` is the chat
// workspace (F3); `/mission` the EU-sovereignty narrative (F2); the auth +
// account routes (F4) follow, with /account guarded.
export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <div className="d-flex flex-column min-vh-100">
            <Header />
            <Routes>
              <Route path="/" element={<Chat />} />
              <Route path="/mission" element={<Mission />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/verify" element={<VerifyEmail />} />
              <Route path="/forgot" element={<RequestReset />} />
              <Route path="/reset" element={<ResetPassword />} />
              <Route
                path="/account"
                element={
                  <RequireAuth>
                    <Dashboard />
                  </RequireAuth>
                }
              />
              <Route
                path="/account/keys"
                element={
                  <RequireAuth>
                    <ApiKeys />
                  </RequireAuth>
                }
              />
            </Routes>
            <Footer />
          </div>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}
