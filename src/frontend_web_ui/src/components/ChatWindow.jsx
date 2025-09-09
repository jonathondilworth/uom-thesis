import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function ChatWindow({
  selectedLLM,
  selectedRetrieval,
  selectedScoreFunction,
  isLoading,
  setIsLoading,
}) {
  
  // State Management for LLM chat and input
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Hi! Ask me a medical question.', fromServer: false },
  ]);
  const [input, setInput] = useState('');


  // State Management for RAG context
  const [fullPromptForModal, setFullPromptForModal] = useState('');
  const [ctxForModal, setCtxForModal] = useState('');
  const [openFull, setOpenFull] = useState(false);
  const [openCtx, setOpenCtx] = useState(false);

  // 
  const addMessage = (role, text, fromServer) => setMessages((prev) => [...prev, { role, text, fromServer }]);

  const handleSend = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim()) return;

    // push user bubble
    addMessage('user', input.trim(), false);
    // clear input
    setInput('');
    // set loading state
    setIsLoading(true);

    // sends request to http://localhost:8000/question
    // update this to be more sophisticated on the FE & BE [TODO]
    try {
      const { data } = await axios.post(
        'http://localhost:8000/question',
        {
          question: input.trim().toLocaleLowerCase(),
          retrieval_method: selectedRetrieval.toLowerCase(),
          // score_function: scoreFn.toLowerCase(),
          // top_k: topK,
          score_function: selectedScoreFunction.toLowerCase()
        },
        { timeout: 180000 } // 3 minutes
      );
      // console log the response (sanity check)
      console.log(data);
      // assumes fullPrompt and retrieved are formatted on the backend and included within the response
      const fullPrompt = data?.full_prompt || data?.prompt || '(no prompt context available]';
      const retrieved = data?.retrieved_context || data?.context || '(no context returned)';
      setFullPromptForModal(`${fullPrompt}`);
      setCtxForModal(retrieved);
      addMessage(
        'ai',
        data?.generated_text || data?.answer || 'Unable to fetch the response.',
        true
      );
    } catch (err) {
      console.error(err);
      alert('QA failed! See console for details...');
    } finally {
      setIsLoading(false);
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // autoscroll
  const bottomRef = useRef(null);
  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), [messages]);
  const lastIndex = messages.length - 1;

  return (
    <div className="flex flex-col h-144 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800">
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, idx) => {
          const isAI = m.role === 'ai';
          const isLastAI = isAI && idx === lastIndex && m.fromServer;
          return (
            <div key={idx} className={`max-w-xs sm:max-w-sm ${isAI ? 'ml-auto' : 'mr-auto'}`}>
              
              <div
                className={`px-4 py-2 rounded-lg whitespace-pre-wrap ${
                  isAI
                    ? 'bg-gray-200 dark:bg-gray-700 dark:text-gray-100'
                    : 'bg-indigo-600 text-white dark:bg-indigo-500'
                }`}
              >
                {m.text}
              </div>

              {isLastAI && (
                <InfoCards
                  onOpenFull={() => setOpenFull(true)}
                  onOpenCtx={() => setOpenCtx(true)}
                />
              )}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={handleSend}
        className="border-t border-gray-300 dark:border-gray-700 p-3 flex gap-2"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Type your message..."
          className="flex-1 resize-none rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 p-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="shrink-0 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md px-4 py-2 text-sm font-medium disabled:opacity-40"
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </form>

      <Modal
        open={openFull}
        title="Full Response (Prompt + Output)"
        onClose={() => setOpenFull(false)}
      >
        {fullPromptForModal}
      </Modal>

      <Modal
        open={openCtx}
        title="Retrieved Context"
        onClose={() => setOpenCtx(false)}
      >
        {ctxForModal}
      </Modal>
    </div>
  );
}

function Modal({ open, title, children, onClose }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />
      
      <div className="relative z-10 w-[90vw] max-w-2xl max-h-[80vh] rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-lg">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
          <h3 className="text-sm font-medium text-gray-800 dark:text-gray-100">{title}</h3>
          <button
            onClick={onClose}
            className="rounded px-2 py-1 text-sm text-gray-500 hover:text-gray-800 dark:hover:text-gray-200"
          >
            Close
          </button>
        </div>
        <div className="p-4 overflow-auto text-sm text-gray-800 dark:text-gray-100 whitespace-pre-wrap">
          {children}
        </div>
      </div>

    </div>
  );
}

function InfoCards({ onOpenFull, onOpenCtx }) {
  return (
    <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
      <button
        onClick={onOpenFull}
        className="text-left rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800 px-3 py-2 shadow-sm"
      >
        <div className="text-xs font-medium text-gray-700 dark:text-gray-200">View Full Response</div>
        <div className="text-[11px] text-gray-500 dark:text-gray-400">Prompt + full model output</div>
      </button>
      <button
        onClick={onOpenCtx}
        className="text-left rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800 px-3 py-2 shadow-sm"
      >
        <div className="text-xs font-medium text-gray-700 dark:text-gray-200">Show Retrieved Context</div>
        <div className="text-[11px] text-gray-500 dark:text-gray-400">Chunks used to answer</div>
      </button>
    </div>
  );
}