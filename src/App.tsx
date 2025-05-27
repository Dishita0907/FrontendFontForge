import React from 'react';
import { FontGenerator } from './components/FontGenerator';

function App() {
  return (
    <div className="min-h-screen bg-paper">
      <header className="bg-white shadow-paper py-6">
        <div className="max-w-4xl mx-auto px-6">
          <h1 className="text-4xl font-display font-bold text-ink">FontForge</h1>
          <p className="text-neutral-600 mt-2">Handwriting Font Generator</p>
        </div>
      </header>
      
      <main className="py-12">
        <FontGenerator />
      </main>
    </div>
  );
}

export default App;