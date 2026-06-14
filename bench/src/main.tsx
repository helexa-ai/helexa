import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import App from "./App";
import Overview from "./pages/Overview";
import Trends from "./pages/Trends";
import Runs from "./pages/Runs";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Overview />} />
          <Route path="trends" element={<Trends />} />
          <Route path="runs" element={<Runs />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
