import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import About from './pages/About';
import Demo from './pages/Demo';
import Search from './Pages/Search';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="about" element={<About />} />
        <Route path="demo" element={<Demo />} />
        <Route path="search" element={<Search />} />
      </Route>
    </Routes>
  );
}