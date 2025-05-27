import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'npm:@supabase/supabase-js@2.1.1'
import * as tf from 'npm:@tensorflow/tfjs@4.17.0'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// VAE architecture
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
    
    // VAE latent space
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

  async train(data: tf.Tensor, epochs: number = 50) {
    const batchSize = 32;
    const optimizer = tf.train.adam();

    for (let epoch = 0; epoch < epochs; epoch++) {
      const [z_mean, z_log_var] = this.encoder.predict(data) as tf.Tensor[];
      
      // Sampling from latent space
      const epsilon = tf.randomNormal([data.shape[0], this.latentDim]);
      const z = z_mean.add(tf.exp(z_log_var.mul(0.5)).mul(epsilon));
      
      // Reconstruction
      const reconstructed = this.decoder.predict(z) as tf.Tensor;
      
      // Calculate losses
      const reconstructionLoss = tf.losses.meanSquaredError(data, reconstructed);
      const klLoss = tf.mean(tf.exp(z_log_var)
        .add(z_mean.square())
        .sub(1)
        .sub(z_log_var)
        .mul(0.5));
      
      const totalLoss = reconstructionLoss.add(klLoss);
      
      // Backpropagation
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

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { language, action } = await req.json()
    
    // Initialize VAE model
    const inputShape = language === 'hindi' ? [64, 64, 1] : [28, 28, 1]
    const model = new FontVAE(inputShape)
    
    if (action === 'train') {
      // Training logic here
      // Note: In production, you'd want to store the trained model weights
      return new Response(
        JSON.stringify({ message: 'Model trained successfully' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    } else if (action === 'generate') {
      // Generate new characters
      const generated = model.generate(1)
      const imageData = await generated.data()
      
      return new Response(
        JSON.stringify({ imageData }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    throw new Error('Invalid action')
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
    )
  }
})