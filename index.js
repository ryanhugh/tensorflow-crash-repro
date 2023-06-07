import fs from 'fs'
const tf = require('@tensorflow/tfjs-node');
import _ from 'lodash';
import sharp from 'sharp';

import * as faceapi from '@vladmandic/face-api';
import tracer from "dd-trace";

import { FaceDetection } from '@vladmandic/face-api';
import imghash from 'imghash';
import * as  nsfw from 'nsfwjs';
import exif from 'exif';
import Tesseract from 'tesseract.js';
import { Pool } from 'pg'
import { QueryLike } from 'sql';

import path from 'path';

tracer.init({
  profiling: true,
  logInjection: true
}); 


// Convert a Node.js Buffer of an image to a TF Tensor object. I copied this code off the internet. 
export function bufferToTensor(buffer) {
  const tensor = tf.tidy(() => {
    const decode = tf.node.decodeImage(buffer, 3);
    let expand;
    if (decode.shape[2] === 4) { // input is in rgba format, need to convert to rgb
      const channels = tf.split(decode, 4, 2); // tf.split(tensor, 4, 2); // split rgba to channels
      const rgb = tf.stack([channels[0], channels[1], channels[2]], 2); // stack channels back to rgb and ignore alpha
      expand = tf.reshape(rgb, [1, decode.shape[0], decode.shape[1], 3]); // move extra dim from the end of tensor and use it as batch number instead
    } else {
      expand = tf.expandDims(decode, 0);
    }
    const cast = tf.cast(expand, 'float32');
    return cast;
  });

  return tensor;
}


async function processWithSharp(buffer) {
    const image = await sharp(buffer)
    await image.metadata()
}


async function init() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join('./faceDetectionModel'))
}

let initPromise;

function getInitPromise() {
  if (!initPromise) {
    initPromise = init();
  }

  return initPromise;
}

let faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.01 });


async function peopleDetection(buffer) {

  await getInitPromise();

  let tensor = bufferToTensor(buffer);
  await faceapi.detectAllFaces(tensor, faceDetectionOptions);

  // TensorFlow requires that we manually manage memory(!!!). 
  // free() the tensor here. 
  tensor.dispose();
}



async function loadModel() {

  let path = 'file://' + process.cwd() + '/nsfwModel/'

  return await nsfw.load(path, {size: 299}) 
}

let modelPromise = null;

function getModel() {
  if (modelPromise) {
    return modelPromise;
  }
  return modelPromise = loadModel();
}


async function nsfwDetection(buffer) {

  let model = await getModel();

  // Image must be in tf.tensor3d format
  // you can convert image to tf.tensor3d with tf.node.decodeImage(Uint8Array,channels)
  const image = await tf.node.decodeImage(buffer, 3)
  await model.classify(image)
  image.dispose() // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
}


function getExif(buffer) {
  return new Promise((resolve, reject) => {
    new exif.ExifImage(buffer, () => {
      resolve(null);
    })
  })
}


async function runOnce() {
    let buffer = fs.readFileSync('./crashy.jpeg', {encoding:null})


    console.log("TF backend:", tf.getBackend())


    await processWithSharp(buffer);
    console.log("got width and height", tf.memory());

    // fs.readFileSync('./fdas', {encoding: null})
    await peopleDetection(buffer);
    console.log('ran people', tf.memory());
    await nsfwDetection(buffer);
    console.log('ran nsfw', tf.memory())
    await imghash.hash(buffer, 8, 'binary')
    console.log('ran hash', tf.memory())
    await Tesseract.recognize(buffer);
    console.log('ran ocr', tf.memory())
    await getExif(buffer);
    console.log('ran exif ', tf.memory())

    console.log('all done')

}



async function main () {
  while (1) {
    await runOnce();
  }
}

if (require.main === module) {
  main()
}