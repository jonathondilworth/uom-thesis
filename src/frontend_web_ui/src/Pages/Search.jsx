import SearchControls from '../components/SearchControls';
import { Divider } from '@/components/catalyst/divider';
import SearchResults from '../components/SearchResults';
import { useState } from 'react';

export default function Search() {

  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col md:flex-row gap-8">
      {/* left column – controls */}
      <aside className="md:w-80 shrink-0">
        <SearchControls 
          setResults={setResults}
          setIsLoading={setIsLoading}
          isLoading={isLoading}
        />
      </aside>

      {/* right column – results placeholder */}
      <section className="flex-1">
        <h2 className="text-xl font-semibold mb-4 dark:text-gray-100">
          Results
        </h2>
        <Divider />

        <SearchResults results={results} isLoading={isLoading} />
      </section>
    </div>
  );
}