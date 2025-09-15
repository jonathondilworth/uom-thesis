export default function Footer() {
  return (
    <footer className="bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 py-6 text-sm flex flex-col sm:flex-row justify-between items-center gap-2">
        <p className="text-gray-600 dark:text-gray-400">
          Knowledge Retrieval Demo
        </p>
        <div className="flex space-x-4 text-inherit dark:text-white">
          <div>
            UI built with:
          </div>
          <a href="#">React</a> <a href="#">TailwindCSS</a> <a href="#">Catalyst</a>
        </div>
      </div>
    </footer>
  );
}