import React from "react";
import "./home.css";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="container full-page">

      <section className="hero full-width">
        <div className="hero-content">
          <h2>The Future of Sign Language Conversion</h2>
          <p>Experience seamless AI-powered sign-to-speech and speech-to-sign translation.</p>
          <button className="cta-button">Get Started</button>
        </div>
        {/* <div className="hero-image">
          <img src="/images/ai-avatar.png" alt="AI Avatar" />
        </div> */}
      </section>

      <section className="testi full-width">
        <div className="testi-card">
          <h3>Real-Time Gesture Recognition</h3>
          <p>Advanced AI-driven hand motion analysis.</p>
        </div>
        <div className="testi-card">
          <h3>3D Avatar Interaction</h3>
          <p>Speech-to-sign translation with a lifelike animated avatar.</p>
        </div>
        <div className="testi-card">
          <h3>Personalized AI Training</h3>
          <p>Customize gestures and enhance AI accuracy.</p>
        </div>
      </section>
    </div>
  );
};

export default Home;
