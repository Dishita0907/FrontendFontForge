import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import fs from 'fs/promises';
import path from 'path';

const HINDI_PATH = './data/hindi';
const ENGLISH_PATH = './data/english';
const OUTPUT_PATH = './data/processed';

async function preprocessImage(imagePath, size) {
  const buffer = await fs.readFile(imagePath);
  const processed = await sharp(buffer)
    .resize(size, size, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
    .grayscale()
    .normalize()
    .toBuffer();

  return tf.node.decodeImage(processed, 1);
}

async function preprocessDataset(inputPath, outputPath, size) {
  await fs.mkdir(outputPath, { recursive: true });
  
  const classes = await fs.readdir(inputPath);
  const processedImages = [];
  const labels = [];

  for (const className of classes) {
    const classPath = path.join(inputPath, className);
    const stats = await fs.stat(classPath);
    
    if (!stats.isDirectory()) continue;

    const images = await fs.readdir(classPath);
    for (const image of images) {
      if (!image.match(/\.(jpg|jpeg|png)$/i)) continue;
      
      const imagePath = path.join(classPath, image);
      const tensor = await preprocessImage(imagePath, size);
      processedImages.push(tensor);
      labels.push(parseInt(className));
    }
  }

  const dataset = tf.stack(processedImages);
  await dataset.save(`file://${outputPath}/dataset`);
  
  const labelsTensor = tf.tensor1d(labels, 'int32');
  await labelsTensor.save(`file://${outputPath}/labels`);
}

async function main() {
  try {
    console.log('Preprocessing Hindi dataset...');
    await preprocessDataset(HINDI_PATH, path.join(OUTPUT_PATH, 'hindi'), 64);
    
    console.log('Preprocessing English dataset...');
    await preprocessDataset(ENGLISH_PATH, path.join(OUTPUT_PATH, 'english'), 28);
    
    console.log('Preprocessing complete!');
  } catch (error) {
    console.error('Error during preprocessing:', error);
    process.exit(1);
  }
}

main();