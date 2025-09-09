import { useState } from 'react';
import ChatComboBox from './ChatComboBox';
import { Field, Label } from '@/components/catalyst/fieldset';
import { Input } from '@/components/catalyst/input';
import { Switch } from '@/components/catalyst/switch';
import axios from 'axios';

export default function SearchControls({setResults, setIsLoading, isLoading}) {

  // options (static)
  const retrievalMethods = ['SBERT', 'HiT', 'OnT'];
  const scoringFunctions = ['Cosine-Similarity', 'Hyperbolic', 'Entity-Subsumption', 'Concept-Subsumption'];

  // state management
  const [query, setQuery] = useState('');
  const [method, setMethod] = useState(retrievalMethods[0]);
  const [scoreFn, setScoreFn] = useState(scoringFunctions[0]);
  const [topK, setTopK] = useState(10);

  // advanced panel state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [weight, setWeight] = useState('0.35');

  // async send request
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResults([]);
    if (!Number.isFinite(weight)) {
      setWeight(0.35)
    }
    try {
      const { data } = await axios.post(
        'http://localhost:8000/search',
        {
          query,
          retrieval_method: method.toLowerCase(),
          score_function: scoreFn.toLowerCase(),
          top_k: topK,
          weight: weight
        },
        { timeout: 60000 }
      );
      console.log(data)
      // sort
      const sorted = [...data].sort((a, b) => a.rank - b.rank);
      setResults(sorted);
    } catch (err) {
      console.error(err);
      alert('Search failed. See console for details.');
    } finally {
      setIsLoading(false);
    }
  };

  // front-end UI form \w options (for search)
  return (
    <form onSubmit={handleSubmit} className="space-y-6">

      <Field>
        <Label>Keyword search</Label>
        <Input
          type="text"
          name="query"
          placeholder="e.g. knee pain"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
      </Field>

      <ChatComboBox
        htmlName="retrieval_method"
        htmlLabel="Retrieval method"
        selectedOption={method}
        opts={retrievalMethods}
        setState={setMethod}
        disabled={isLoading}
      />

      <ChatComboBox
        htmlName="scoring_function"
        htmlLabel="Score function"
        selectedOption={scoreFn}
        opts={scoringFunctions}
        setState={setScoreFn}
        disabled={isLoading}
      />

      <Field>
        <Label>Fetch top-K</Label>
        <Input
          type="number"
          name="topk"
          min={1}
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value, 10) || 1)}
          disabled={isLoading}
        />
      </Field>

      <div className="flex items-center justify-between">
        <span className="text-sm font-medium dark:text-white">Advanced options</span>
        <Switch checked={showAdvanced} onChange={setShowAdvanced} disabled={isLoading} />
      </div>

      {showAdvanced && (
        <div className="space-y-4 border-t pt-4 dark:text-white">
        <Field>
          <Label>Centripetal Weight (Depth Bias)</Label>
          <Input
            type="number"
            inputMode="decimal"
            step="0.01"
            min="0"
            max="5"
            name="centripetal_weight"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            disabled={isLoading}
          />
        </Field>
        </div>
      )}

      <button
        className="catalyst-button primary w-full disabled:opacity-40 dark:text-white"
        disabled={isLoading || !query.trim()}
      >
        {isLoading ? 'Searching...' : 'Search'}
      </button>
    </form>
  );
}