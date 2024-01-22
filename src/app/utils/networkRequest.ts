import { post } from "@/app/utils/request";
import { IShapeParams, IArchitecture, ITraining, IBuildParams, IRunParams, IBuildRes, IRunRes } from "@/app/types/INetwork";

export function buildShape(shapeParams: IShapeParams) {
  const fullShape: number[] = [];
  fullShape.push(shapeParams.inputShape);
  shapeParams.hiddenShape.forEach(n => fullShape.push(n));
  fullShape.push(shapeParams.outputShape);
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