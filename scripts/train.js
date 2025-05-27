import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs/promises';
import path from 'path';

class FontVAE {
  constructor(inputShape, latentDim = 128) {
    this.inputShape = inputShape;
    this.latentDim = latentDim;
    this.buildModel();
  }

  buildModel() {
    // Encoder
    const inputs = tf.input({ shape: this.inputShape });
    let x = inputs;

    // Convolutional layers with batch normalization and dropout
    const encoderFilters = [32, 64, 128, 256];
    for (const filters of encoderFilters) {
      x = tf.layers.conv2d({
        filters,
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: 'relu'
      }).apply(x);
      x = tf.layers.batchNormalization().apply(x);
      x = tf.layers.dropout({ rate: 0.2 }).apply(x);
    }

    const flat = tf.layers.flatten().apply(x);
    
    // VAE latent space
    this.z_mean = tf.layers.dense({ units: this.latentDim }).apply(flat);
    this.z_log_var = tf.layers.dense({ units: this.latentDim }).apply(flat);
    
    // Sampling layer
    const z = tf.layers.lambda((tensors) => {
      const [mean, logvar] = tensors;
      const epsilon = tf.randomNormal([mean.shape[0], this.latentDim]);
      return mean.add(tf.exp(logvar.mul(0.5)).mul(epsilon));
    }).apply([this.z_mean, this.z_log_var]);

    // Decoder
    let decoderShape = this.calculateDecoderShape();
    let y = tf.layers.dense({
      units: decoderShape.reduce((a, b) => a * b),
      activation: 'relu'
    }).apply(z);
    y = tf.layers.reshape({ targetShape: decoderShape }).apply(y);

    // Transposed convolutions with batch normalization
    const decoderFilters = [128, 64, 32, 1];
    for (let i = 0; i < decoderFilters.length; i++) {
      const isLast = i === decoderFilters.length - 1;
      y = tf.layers.conv2dTranspose({
        filters: decoderFilters[i],
        kernelSize: 3,
        strides: 2,
        padding: 'same',
        activation: isLast ? 'sigmoid' : 'relu'
      }).apply(y);
      
      if (!isLast) {
        y = tf.layers.batchNormalization().apply(y);
      }
    }

    // Build models
    this.encoder = tf.model({ inputs, outputs: [this.z_mean, this.z_log_var] });
    this.decoder = tf.model({ inputs: z, outputs: y });
    this.vae = tf.model({ inputs, outputs: y });
  }

  calculateDecoderShape() {
    const [height, width, channels] = this.inputShape;
    const reduction = Math.pow(2, 4); // 4 conv layers with stride 2
    return [
      Math.ceil(height / reduction),
      Math.ceil(width / reduction),
      256
    ];
  }

  async train(dataset, epochs = 100, batchSize = 32) {
    const optimizer = tf.train.adam({ learningRate: 0.0001 });
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      const losses = [];
      
      for (let batch of dataset.batch(batchSize)) {
        const loss = optimizer.minimize(() => {
          const [z_mean, z_log_var] = this.encoder.predict(batch);
          const reconstructed = this.decoder.predict(z_mean);
          
          const reconstructionLoss = tf.losses.meanSquaredError(batch, reconstructed);
          const klLoss = tf.mean(
            tf.exp(z_log_var)
              .add(z_mean.square())
              .sub(1)
              .sub(z_log_var)
              .mul(0.5)
          );
          
          return reconstructionLoss.add(klLoss.mul(0.1));
        }, true);
        
        losses.push(loss.dataSync()[0]);
      }
      
      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}: Average Loss = ${losses.reduce((a, b) => a + b) / losses.length}`);
      }
    }
  }

  async save(modelPath) {
    await this.vae.save(`file://${modelPath}`);
  }

  async load(modelPath) {
    this.vae = await tf.loadLayersModel(`file://${modelPath}/model.json`);
  }

  generate(n = 1) {
    const z = tf.randomNormal([n, this.latentDim]);
    return this.decoder.predict(z);
  }
}

async function trainModel(language) {
  const dataPath = path.join('./data/processed', language);
  const modelPath = path.join('./models', language);
  
  // Load preprocessed dataset
  const dataset = await tf.data.array(
    await tf.loadLayersModel(`file://${dataPath}/dataset/model.json`)
  );
  
  // Initialize and train model
  const inputShape = language === 'hindi' ? [64, 64, 1] : [28, 28, 1];
  const model = new FontVAE(inputShape);
  
  console.log(`Training ${language} model...`);
  await model.train(dataset);
  
  // Save trained model
  await fs.mkdir(modelPath, { recursive: true });
  await model.save(modelPath);
  
  console.log(`${language} model training complete!`);
}

async function main() {
  try {
    await trainModel('hindi');
    await trainModel('english');
  } catch (error) {
    console.error('Error during training:', error);
    process.exit(1);
  }
}

main();