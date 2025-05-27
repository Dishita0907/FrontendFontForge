import * as tf from '@tensorflow/tfjs';

export interface FontGeneratorResponse {
  imageData?: number[];
  message?: string;
  error?: string;
}

class FontVAE {
  private encoder: tf.LayersModel;
  private decoder: tf.LayersModel;
  private latentDim: number;

  constructor(inputShape: number[], latentDim: number = 128) {
    this.latentDim = latentDim;
    
    // Encoder
    const inputs = tf.input({shape: inputShape});
    const x1 = tf.layers.conv2d({filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}).apply(inputs);
    const x2 = tf.layers.conv2d({filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}).apply(x1);
    const x3 = tf.layers.conv2d({filters: 128, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}).apply(x2);
    const flat = tf.layers.flatten().apply(x3);
    
    const z_mean = tf.layers.dense({units: latentDim}).apply(flat);
    const z_log_var = tf.layers.dense({units: latentDim}).apply(flat);
    
    this.encoder = tf.model({inputs: inputs, outputs: [z_mean, z_log_var]});

    // Decoder
    const decoderInputs = tf.input({shape: [latentDim]});
    const x4 = tf.layers.dense({units: 7 * 7 * 128, activation: 'relu'}).apply(decoderInputs);
    const x5 = tf.layers.reshape({targetShape: [7, 7, 128]}).apply(x4);
    const x6 = tf.layers.conv2dTranspose({filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}).apply(x5);
    const x7 = tf.layers.conv2dTranspose({filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu'}).apply(x6);
    const outputs = tf.layers.conv2dTranspose({filters: 1, kernelSize: 3, strides: 2, padding: 'same', activation: 'sigmoid'}).apply(x7);
    
    this.decoder = tf.model({inputs: decoderInputs, outputs: outputs});
  }

  async train(data: tf.Tensor, epochs: number = 50): Promise<void> {
    const batchSize = 32;
    const optimizer = tf.train.adam();

    for (let epoch = 0; epoch < epochs; epoch++) {
      const [z_mean, z_log_var] = this.encoder.predict(data) as tf.Tensor[];
      
      const epsilon = tf.randomNormal([data.shape[0], this.latentDim]);
      const z = z_mean.add(tf.exp(z_log_var.mul(0.5)).mul(epsilon));
      
      const reconstructed = this.decoder.predict(z) as tf.Tensor;
      
      const reconstructionLoss = tf.losses.meanSquaredError(data, reconstructed);
      const klLoss = tf.mean(tf.exp(z_log_var)
        .add(z_mean.square())
        .sub(1)
        .sub(z_log_var)
        .mul(0.5));
      
      const totalLoss = reconstructionLoss.add(klLoss);
      
      const grads = tf.grads(() => totalLoss);
      optimizer.applyGradients(grads);
      
      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}: Loss = ${totalLoss.dataSync()[0]}`);
      }
    }
  }

  generate(n: number = 1): tf.Tensor {
    const z = tf.randomNormal([n, this.latentDim]);
    return this.decoder.predict(z) as tf.Tensor;
  }
}

let model: FontVAE | null = null;

export const generateFont = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  try {
    if (!model) {
      const inputShape = language === 'hindi' ? [64, 64, 1] : [28, 28, 1];
      model = new FontVAE(inputShape);
    }

    const generated = model.generate(1);
    const imageData = await generated.data();
    
    return { imageData: Array.from(imageData) };
  } catch (error) {
    return { error: error.message };
  }
};

export const trainModel = async (language: 'hindi' | 'english'): Promise<FontGeneratorResponse> => {
  try {
    const inputShape = language === 'hindi' ? [64, 64, 1] : [28, 28, 1];
    model = new FontVAE(inputShape);
    
    // For demo purposes, we're using random data
    // In production, you'd load and preprocess your dataset here
    const dummyData = tf.randomNormal([100, ...inputShape]);
    await model.train(dummyData);
    
    return { message: 'Model trained successfully' };
  } catch (error) {
    return { error: error.message };
  }
};