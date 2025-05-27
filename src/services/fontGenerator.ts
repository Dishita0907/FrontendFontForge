import { SUPABASE_URL, SUPABASE_ANON_KEY } from '../config';

export interface FontGeneratorResponse {
  imageData?: number[];
  message?: string;
  error?: string;
}

export const generateFont = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  try {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/font-generator`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        language,
        action: 'generate'
      })
    });

    if (!response.ok) {
      throw new Error('Failed to generate font');
    }

    return await response.json();
  } catch (error) {
    return { error: error.message };
  }
};

export const trainModel = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  try {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/font-generator`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        language,
        action: 'train'
      })
    });

    if (!response.ok) {
      throw new Error('Failed to train model');
    }

    return await response.json();
  } catch (error) {
    return { error: error.message };
  }
};