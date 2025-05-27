import * as tf from '@tensorflow/tfjs';

export interface FontGeneratorResponse {
  imageData?: number[];
  message?: string;
  error?: string;
}

class FontVAE {
  private vae: tf.LayersModel | null = null;
  private decoder: tf.LayersModel | null = null;
  private latentDim: number = 128;

  async load(language: string) {
    try {
      this.vae = await tf.loadLayersModel(`/models/${language}/model.json`);
      this.decoder = this.vae.layers[this.vae.layers.length - 1];
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error('Failed to load model');
    }
  }

  generate(n: number = 1): tf.Tensor {
    if (!this.decoder) {
      throw new Error('Model not loaded');
    }
    
    const z = tf.randomNormal([n, this.latentDim]);
    return this.decoder.predict(z) as tf.Tensor;
  }
}

const models: Record<string, FontVAE> = {
  hindi: new FontVAE(),
  english: new FontVAE()
};

export const generateFont = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  try {
    if (!models[language].vae) {
      await models[language].load(language);
    }

    const generated = models[language].generate(1);
    const imageData = await generated.data();
    
    return { imageData: Array.from(imageData) };
  } catch (error) {
    return { error: error.message };
  }
};

export const trainModel = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  return { message: 'Training is handled by the backend scripts' };
};