import React from 'react';

export default function About() {
  return (
     <article className="prose prose-invert mx-auto max-w-3xl px-6 py-6">
      <h1 className="mb-4 text-inherit dark:text-white">README.md goes here...</h1>
      <p>
        README.md goes here...
      </p>

      <h2 className="text-inherit dark:text-white">Tech stack &amp; Libs</h2>
      <ul>
        <li><strong className="text-inherit dark:text-white">Front-end:</strong> Vite, React, TailwindCSS, Catalyst, <em>(HTML, CSS, JavaScript)</em></li>
        <li><strong className="text-inherit dark:text-white">Back-end:</strong> FastAPI, Pydantic, <em>(Python)</em></li>
        <li><strong className="text-inherit dark:text-white">ML/DL:</strong> PyTorch, SpaCy, SciSpaCy, BM25Okapi, SentenceTransformer, HF Transformer Libs, HiT, OnT</li>
        <li><strong className="text-inherit dark:text-white">LLMs:</strong> Mistral-7B</li>
      </ul>

      <h2 className="text-inherit dark:text-white">Roadmap</h2>
      <ol>
        <li><s>Review OnT.py &amp; ELNormalizeData.py, log instances where previously ran into runtime errors.</s></li>
        <li><s>Train OnT &amp; OnTr models on most recent release of SNOMED CT</s></li>
        <li><s>Fix hyperbolic distance ranking.</s></li>
        <li><s>Curate dataset for measuring retrieval performance.</s></li>
        <li>Implement advanced retrieval mechanisms. <em>(optional?)</em></li>
        <li>Add weighted score function <em>(or utilise Product Manifolds?)</em><br />
          <ul>
            <li>See: <a href="https://openreview.net/pdf?id=HJxeWnCcF7">LEARNING MIXED-CURVATURE REPRESENTATIONS IN PRODUCTS OF MODEL SPACES</a></li>
          </ul>
        </li>
        <li>Include parameter adjustment UI</li>
        <li><s>Implement Subsumption Retrieval</s></li>
        <li><s>Implement RAG pipeline</s></li>
      </ol>

      <p className="mt-10 text-sm opacity-70">
        MIT License
      </p>
    </article>
  );
}