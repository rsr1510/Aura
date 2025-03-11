import React from "react";
import "./header.css";
import { Link } from "react-router-dom";

const Header = () => {
    return (
        <header className="navbar">
            <div className="logo">Aura</div>

            <div className="link">
                <Link to={"/"}>Home</Link>
                <Link to={"/features"}>Features</Link>
                <Link to={"/about"}>About</Link>
                <Link to={"/contact"}>Contact</Link>
            </div>
        </header>
    );
};

export default Header;
