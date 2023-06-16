const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { NSFWJS } = require('nsfwjs');
let faceapi = require('@vladmandic/face-api')
const path = require("path");
let sharp = require('sharp')
let imghash = require('imghash')
let Tesseract = require('tesseract.js')
let exif = require('exif')
require('pg')
require('sql')


// Convert a Node.js Buffer of an image to a TF Tensor object. I copied this code off the internet.
function bufferToTensor(buffer) {
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

function getExif(buffer) {
  return new Promise((resolve, reject) => {
    new exif.ExifImage(buffer, () => {
      resolve(null);
    })
  })
}


async function loadNsfwModel() {
  const modelPath = './nsfwModel/model.json';
  const modelURL = 'file://' + modelPath;

  const nsfw = new NSFWJS(0, { size: 299 });
  nsfw.load = async function() {
    this.model = await tf.loadLayersModel(modelURL);
  };
  await nsfw.load();
  return nsfw;
}

async function peopleDetection(buffer) {
  let faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.01 });

  let tensor = bufferToTensor(buffer);
  await faceapi.detectAllFaces(tensor, faceDetectionOptions);

  // TensorFlow requires that we manually manage memory(!!!).
  // free() the tensor here.
  await tensor.dispose();

  // deleting buffer
  buffer = null
}

async function nsfwDetection(buffer, model) {
  const imageTensor = tf.node.decodeImage(buffer);
  const image = imageTensor.expandDims();

  await model.classify(image);

  await image.dispose();
  await imageTensor.dispose();
}

async function runOnce(buffer, model) {
  await tf.setBackend('tensorflow');
  console.log("TF backend:", tf.getBackend());

  await processWithSharp(buffer);
  console.log("got width and height", tf.memory());

  await nsfwDetection(buffer, model);
  console.log('ran nsfw', tf.memory());

  await peopleDetection(buffer);
  console.log('ran people', tf.memory());

  await imghash.hash(buffer, 8, 'binary')
  console.log('ran hash', tf.memory())

  await Tesseract.recognize(buffer);
  console.log('ran ocr', tf.memory())

  await getExif(buffer);
  console.log('ran exif ', tf.memory())

  //await tf.nextFrame()

}



async function runOnceParallel(buffer, model) {
  await tf.setBackend('tensorflow');
  console.log("TF backend:", tf.getBackend());

  console.log('running all at once')
  await Promise.all([
    processWithSharp(buffer),
    nsfwDetection(buffer, model),
    peopleDetection(buffer),
    imghash.hash(buffer, 8, 'binary'),
    Tesseract.recognize(buffer),
    getExif(buffer)

  ])
}




let CWD = process.env.MY_CWD || process.cwd() ;


async function main() {
  const buffer = fs.readFileSync(CWD + '/crashy.jpeg', { encoding: null });
  const model = await loadNsfwModel();
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(CWD + '/faceDetectionModel'))

  let i = 0;
  while (true) {
    await runOnceParallel(buffer, model);
    i += 1;
    console.log("ran", i, "times");
    if (i === 100) {
      break;
    }
  }
}

if (require.main === module) {
  main();
}
