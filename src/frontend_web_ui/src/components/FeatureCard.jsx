import React from 'react';

export default function FeatureCard({ icon, title, description }) {
  return (
    <div className="flex flex-col items-start p-6 bg-white dark:bg-gray-800 rounded-xl shadow">
      <div className="mb-4">
        <div className="flex items-center justify-center h-12 w-12 rounded-full bg-indigo-50 dark:bg-indigo-900">
          {icon}
        </div>
      </div>
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100 mb-2">{title}</h3>
      <p className="text-gray-700 dark:text-gray-300">{description}</p>
    </div>
  );
}