import React from 'react';

export default function AdvancedPanel(props) {
  const {
    temperature, setTemperature,
    topK, setTopK
  } = props;

  return (
    <div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Temperature</label>
        <div className="flex items-center space-x-3">
          <input 
            type="range" min="0" max="1" step="0.1" 
            value={temperature} 
            onChange={e => setTemperature(parseFloat(e.target.value))} 
            className="flex-1"
          />
          <span className="text-sm text-gray-800 dark:text-gray-200">{temperature.toFixed(1)}</span>
        </div>
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Top-K</label>
        <input 
          type="number" min="1" max="100" 
          value={topK} 
          onChange={e => setTopK(parseInt(e.target.value) || 0)} 
          className="w-full rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 p-2 text-sm"
        />
      </div>
      {/* TODO: Use these advanced settings in the generation logic when ready/in future */}
    </div>
  );
}