import React from 'react';
import HeroSection from '../components/HeroSection';
import FeatureCard from '../components/FeatureCard';
import { AcademicCapIcon, AdjustmentsVerticalIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

export default function Home() {
  return (
    <>
      <HeroSection />
      <section className="py-12">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-8">Key Features</h2>
          <div className="grid gap-8 md:grid-cols-3">
            <FeatureCard 
              title="Intelligent Retrieval" 
              description="Whilst naming conceptualisations is powerful for identifying and communicating concepts, don't forget about structure!"
              icon={<MagnifyingGlassIcon className="h-6 w-6 text-indigo-600 dark:text-indigo-200" aria-hidden="true" />} 
            />
            <FeatureCard 
              title="Knowledge-Aware" 
              description=" Move beyond distributional semantics, embrase retrieval using structural properties of formal knowledge representations!" 
              icon={<AcademicCapIcon className="h-6 w-6 text-indigo-600 dark:text-indigo-200" aria-hidden="true" />} 
            />
            <FeatureCard 
              title="Configurable Generation" 
              description="Select from a cornucopia of configuration options in your hunt for effective knowledge retrieval!" 
              icon={<AdjustmentsVerticalIcon className="h-6 w-6 text-indigo-600 dark:text-indigo-200" aria-hidden="true" />} 
            />
          </div>
        </div>
      </section>
    </>
  );
}