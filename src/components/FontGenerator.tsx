import React, { useState } from 'react';
import { generateFont, trainModel } from '../services/fontGenerator';
import type { FontGeneratorResponse } from '../services/fontGenerator';

export const FontGenerator: React.FC = () => {
  const [language, setLanguage] = useState<'hindi' | 'english'>('english');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<number[] | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    
    const response = await generateFont(language);
    
    if (response.error) {
      setError(response.error);
    } else if (response.imageData) {
      setGeneratedImage(response.imageData);
    }
    
    setLoading(false);
  };

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    
    const response = await trainModel(language);
    
    if (response.error) {
      setError(response.error);
    } else if (response.message) {
      alert(response.message);
    }
    
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-card p-8">
        <h2 className="text-3xl font-display font-bold mb-6">Font Generator</h2>
        
        <div className="mb-6">
          <label className="block text-sm font-semibold mb-2">Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value as 'hindi' | 'english')}
            className="w-full p-2 border rounded-md"
            disabled={loading}
          >
            <option value="english">English</option>
            <option value="hindi">Hindi</option>
          </select>
        </div>

        <div className="flex gap-4 mb-6">
          <button
            onClick={handleTrain}
            disabled={loading}
            className="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
          >
            {loading ? 'Training...' : 'Train Model'}
          </button>
          
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="px-6 py-2 bg-accent-600 text-white rounded-md hover:bg-accent-700 disabled:opacity-50"
          >
            {loading ? 'Generating...' : 'Generate Font'}
          </button>
        </div>

        {error && (
          <div className="text-accent-600 mb-6">
            Error: {error}
          </div>
        )}

        {generatedImage && (
          <div className="border rounded-md p-4">
            <h3 className="text-xl font-semibold mb-4">Generated Character</h3>
            <canvas
              ref={(canvas) => {
                if (canvas) {
                  const ctx = canvas.getContext('2d');
                  if (ctx) {
                    const imageData = new ImageData(
                      new Uint8ClampedArray(generatedImage),
                      language === 'hindi' ? 64 : 28,
                      language === 'hindi' ? 64 : 28
                    );
                    ctx.putImageData(imageData, 0, 0);
                  }
                }
              }}
              width={language === 'hindi' ? 64 : 28}
              height={language === 'hindi' ? 64 : 28}
              className="border"
            />
          </div>
        )}
      </div>
    </div>
  );
};