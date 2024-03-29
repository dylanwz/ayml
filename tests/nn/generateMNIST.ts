import * as fs from 'fs';

function readMNISTCSV(csvFile: string, numSamples: number): { labels: number[], flattenedData: number[][] } {
  const content = fs.readFileSync(csvFile, 'utf-8');
  const lines = content.trim().split('\n');

  const labels: number[] = [];
  const flattenedData: number[][] = [];

  for (let i = 0; i < numSamples; i++) {
    const values = lines[i].split(',').map(Number);
    labels.push(values[0]); // First value is the label
    flattenedData.push(values.slice(1).map((v) => v/255)); // Remaining values are pixel intensities
  }

  return { labels, flattenedData };
}

// Example usage
const numTrainSamples = 10000;
const numTestSamples = 100;

export const { labels: trainLabels, flattenedData: flattenedTrainData } = readMNISTCSV('data/mnist_train.csv', numTrainSamples);
export const { labels: testLabels, flattenedData: flattenedTestData } = readMNISTCSV('data/mnist_test.csv', numTestSamples);

// for (let i = 0; i < flattenedTrainData[0].length; i++) {
//   process.stdout.write(String(flattenedTrainData[0][i]))
//   process.stdout.write(', ')
// }
// console.log(trainLabels[0])