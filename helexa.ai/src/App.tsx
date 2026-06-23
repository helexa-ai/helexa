import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Container } from "react-bootstrap";
import ThemeProvider from "./layout/ThemeProvider";
import Header from "./components/Header";
import Footer from "./components/Footer";
import "./App.css";

// F1 composition root: theme + router + layout shell. The chat workspace
// (`/`, F3), `/mission` (F2), and the auth/account routes (F4) replace these
// placeholders in later phases.
function Placeholder({ title }: { title: string }) {
  return (
    <Container className="py-5 flex-grow-1">
      <h1 className="mb-2">{title}</h1>
      <p className="text-muted">helexa public beta — coming online.</p>
    </Container>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="d-flex flex-column min-vh-100">
          <Header />
          <Routes>
            <Route path="/" element={<Placeholder title="Chat" />} />
            <Route path="/mission" element={<Placeholder title="Mission" />} />
          </Routes>
          <Footer />
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}
