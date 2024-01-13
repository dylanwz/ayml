import { post } from "@/app/utils/request";
import { IArchitecture, ITraining, IBuildParams, IRunParams, IBuildRes, IRunRes } from "@/app/types/INetwork";

export function buildShape(shape: number[], numHiddenLayers: number, numHiddenNeurons: number) {
  const fullShape = [...shape];
  const hiddenShape = Array.from({length: numHiddenLayers}, () => numHiddenNeurons);
  fullShape.splice(1,0,...hiddenShape);
  return fullShape;
}

const voidFunction = (i: any) => {
  return;
}

export async function startNetwork(
  architectureParams: IArchitecture,
  trainingParams: ITraining,
  onFinish?: (res: IBuildRes) => void)
{
  const buildParams: IBuildParams = {"architectureParams": architectureParams, "trainingParams": trainingParams};
  const res: IBuildRes = await post("/classifier/classic/start", buildParams);

  const callback = onFinish ?? voidFunction;
  callback(res);
}

export async function runNetwork(
  serviceID: string, 
  dataspace: string,
  onFinish?: (res: IRunRes) => void)
{
  const runParams: IRunParams = {"serviceID": serviceID, "dataspace": dataspace};
  const res: IRunRes = await post("/classifier/classic/run", runParams);

  const callback = onFinish ?? voidFunction;
  callback(res);
}