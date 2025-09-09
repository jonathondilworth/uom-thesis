import { Table, TableHead, TableHeader, TableRow, TableBody, TableCell } from '@/components/catalyst/table';

import Spinner from './utility_components/Spinner'; // or any spinner component you have

export default function SearchResults({ results, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <Spinner size="lg" />
        <span className="ml-3 text-sm text-gray-600 dark:text-gray-300">Fetching results…</span>
      </div>
    );
  }

  if (!results.length) {
    return (
      <p className="mt-4 text-gray-500 dark:text-gray-400">
        No results yet. Run a search to see documents ranked here.
      </p>
    );
  }

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeader>Rank</TableHeader>
          <TableHeader>ID</TableHeader>
          <TableHeader>Score</TableHeader>
          <TableHeader>Text</TableHeader>
        </TableRow>
      </TableHead>
      <TableBody>
        {results.map(({ rank, id, score, text }) => (
          <TableRow key={id}>
            <TableCell>{rank}</TableCell>
            <TableCell>{id}</TableCell>
            <TableCell>{score}</TableCell>
            <TableCell className="whitespace-pre-wrap">{text}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}