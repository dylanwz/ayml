/**
 * (Majority) From https://playground.tensorflow.org/
 */

type Point = {
  x: number;
  y: number;
}

export type LabelledPoint = {
  x: number;
  y: number;
  label: number;
}

function dist(a: Point | LabelledPoint, b: Point | LabelledPoint) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function generateCircleData(numSamples: number, noise: number): LabelledPoint[] {
  let points: LabelledPoint[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

/**
 * 
 * From https://deeperplayground.org/
 * 
 */
export function generateHeartData(numSamples: number, noise: number): LabelledPoint[] {
  function polarHeart(t: number): number {
    t += Math.PI/2;
    // From https://pavpanchekha.com/blog/heart-polar-coordinates.html
    let r = (Math.sin(t)*Math.sqrt(Math.abs(Math.cos(t)))) / (Math.sin(t) + 7/5) -
      2 * Math.sin(t) + 2;
    return r;
  }
  
  let points: LabelledPoint[] = [];
  let step = (2 * Math.PI) / (numSamples/2);
  for (let i = 0; i < numSamples/2; i++) {
    let t = i * step;
    let r = polarHeart(t);
    let x = 2.25 * r * Math.sin(t) + randUniform(-1, 1) * noise;
    let y = 3.75 + 2.25 * r * Math.cos(t) + randUniform(-1, 1) * noise;
    let label = -1;
    points.push({x, y, label});
    x = 2.25 * (r-0.5) * Math.sin(t) + randUniform(-1, 1) * noise;
    y = 3.5 + 2.25 * (r-0.6) * Math.cos(t) + randUniform(-1, 1) * noise;
    label = 1;
    points.push({x, y, label});
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}