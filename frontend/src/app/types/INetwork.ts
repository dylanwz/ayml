export type IArchitecture = {
  networkShape: number[];
  activation: string;
  outputActivation: string;
  regularisation: string;
  initZero: boolean;
}

export type ITraining = {
  batchSize: number;
  learningRate: number;
  regLambda: number;
  lossFn: string;
}

export type IBuildParams = {
  architectureParams: IArchitecture;
  trainingParams: ITraining;
}

export type IRunParams = {
  serviceID: string;
  dataspace: string;
}

export type IBuildRes = {
  serviceID: string;
}

export type IRunRes = {
  epochs: number;
  loss: number;
}

export type IShapeParams = {
  inputShape: number;
  hiddenShape: number[];
  outputShape: number;
}