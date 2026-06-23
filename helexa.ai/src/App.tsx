import { BrowserRouter, Routes, Route } from "react-router-dom";
import ThemeProvider from "./layout/ThemeProvider";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Mission from "./pages/Mission";
import Chat from "./pages/Chat";
import "./App.css";

// Composition root: theme + router + layout shell. `/` is the chat
// workspace (F3, anonymous for now); `/mission` (F2) is the EU-sovereignty
// narrative; the auth/account routes (F4) land next.
export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="d-flex flex-column min-vh-100">
          <Header />
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/mission" element={<Mission />} />
          </Routes>
          <Footer />
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}
