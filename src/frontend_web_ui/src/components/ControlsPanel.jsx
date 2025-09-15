// const [chunkSize, setChunkSize] = useState(1000);
// const [promptEnhance, setPromptEnhance] = useState(false);

import React, { useState } from 'react';
import ChatComboBox from './ChatComboBox';
// TODO: use AdvancedPanel when application is more mature
import AdvancedPanel from './AdvancedPanel';

export default function ControlsPanel({selectedLLM, selectedRetrieval, selectedScoreFunction, setSelectedScoreFunction, setSelectedLLM, setSelectedRetrieval, isLoading}) {
  
  const LLMOptions = [
    'Mistral-7B-v0.3'
  ]; // TODO: add more language models
  
  // TODO: change this out for a callback
  const retrievalMethods = [
    'HiT',
    'OnT'
  ];

  const scoreFunctions = [
    'hyperbolic'
  ];

  // TODO: implement temperature, top_k, top_p, etc
  return (
    <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
      <div className="mb-4 flex flex-col gap-4">
        <ChatComboBox 
            htmlName="language_model_backbone" 
            htmlLabel="Language Model" 
            selectedOption={selectedLLM}
            opts={LLMOptions} 
            setState={setSelectedLLM}
            disabled={isLoading}
        />
        <ChatComboBox 
            htmlName="retrieval_method" 
            htmlLabel="Retrieval Method"
            selectedOption={selectedRetrieval}
            opts={retrievalMethods} 
            setState={setSelectedRetrieval}
            disabled={isLoading}
        />
        <ChatComboBox 
            htmlName="score_function" 
            htmlLabel="Score Function"
            selectedOption={selectedScoreFunction}
            opts={scoreFunctions} 
            setState={setSelectedScoreFunction}
            disabled={isLoading}
        />
      </div>

    </div>
  );
}