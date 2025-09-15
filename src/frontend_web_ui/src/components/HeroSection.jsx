import React from 'react';
import { Link } from 'react-router-dom';

export default function HeroSection() {
  return (
    <section className="bg-gradient-to-br from-indigo-600 to-purple-600 text-white">
      <div className="max-w-3xl mx-auto text-center py-20 px-4">
        <h1 className="text-4xl font-extrabold mb-4">Information Retrieval &amp; RAG Demo</h1>
        <p className="text-lg mb-6">
          A demo showcasing Lexical, Semantic &amp; Ontological Retrieval for Downstream 
          BioMed MCQA in RAG pipelines <em>for demonstration purposes only</em>.
        </p>
        <Link 
          to="/demo" 
          className="bg-white text-indigo-700 font-medium px-6 py-3 rounded-md shadow hover:bg-gray-50 transition"
        >
          Try it out
        </Link>
      </div>
    </section>
  );
}