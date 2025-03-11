import React from "react";
import "./styles.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/home/home.jsx";
import Features from "./pages/features/features.jsx"
import About from "./pages/about/about.jsx";
import Contact from "./pages/contact/contact.jsx";
import Header from "./components/header/header.jsx";
import Footer from "./components/footer/footer.jsx";


const App = () => {
  return (
    <>
        <BrowserRouter>
          <Header/>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/features" element={<Features />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
          <Footer/>
        </BrowserRouter>
    </>
  );
};

export default App;
