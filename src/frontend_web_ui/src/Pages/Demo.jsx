import ChatWindow from '../components/ChatWindow';
import ControlsPanel from '../components/ControlsPanel';
import { useState } from 'react';

export default function Demo() {

// supported LLMs; swap out for call to /get/llms [TODO]
  const LLMOptions = [
    'Mistral-7B-v0.3' // https://huggingface.co/mistralai/Mistral-7B-v0.1 << 59k downloads last month
    // 'BioMistral-7B', // https://huggingface.co/BioMistral/BioMistral-7B << 43k downloads last month
    // 'Mistral-7B-v0.1' // https://huggingface.co/mistralai/Mistral-7B-v0.3 << 325k downloads last month
  ]; // add more language models
  
  // Swap out for calls to a REST API (/get/retrievers) [TODO]
  const retrievalMethods = [
    'HiT',
    'OnT'
  ];

  // Swap out for call to /get/score-functions [TODO]
  const scoreFunctions = [
    'hyperbolic'
  ];

  // State Management
  const [selectedLLM, setSelectedLLM] = useState(LLMOptions[0]);
  const [selectedRetrieval, setSelectedRetrieval] = useState(retrievalMethods[0]);
  const [selectedScoreFunction, setSelectedScoreFunction] = useState(scoreFunctions[0])
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row gap-8">
        
        <div className="flex-1">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-4">Chat Interface</h2>
          <ChatWindow 
            LLMOptions={LLMOptions}
            retrievalMethods={retrievalMethods}
            scoreFunctions={scoreFunctions}
            selectedLLM={selectedLLM}
            selectedRetrieval={selectedRetrieval}
            selectedScoreFunction={selectedScoreFunction}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>
        
        <div className="md:w-1/4">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-4">Configure RAG</h2>
          <ControlsPanel
            selectedLLM={selectedLLM}
            selectedRetrieval={selectedRetrieval}
            selectedScoreFunction={selectedScoreFunction}
            setSelectedLLM={setSelectedLLM}
            setSelectedRetrieval={setSelectedRetrieval}
            setSelectedScoreFunction={setSelectedScoreFunction}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}