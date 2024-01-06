'use client'

import { useEffect, useState } from "react";
import { ArrowUturnLeftIcon } from "@heroicons/react/24/outline";
import { PlayIcon } from "@heroicons/react/24/solid";
import { PauseIcon } from "@heroicons/react/24/solid";
import { ChevronDoubleRightIcon } from "@heroicons/react/24/outline";
import Dropdown from "../Dropdown/Dropdown";

import { post } from "@/app/utils/request";

export default function ClassicContent() {
  const [start, setStart] = useState(false);
  const [epochs, setEpochs] = useState(0);
  const epochString = ('000000' + epochs).slice(-6);

  const [networkShape, setNetworkShape] = useState([4, 2, 2, 4]);
  const [activation, setActivation] = useState("Tanh");
  const activations = ["ReLU", "Tanh", "Sigmoid", "Linear"];
  const [regularisation, setRegularisation] = useState("None");
  const regularisations = ["None", "L1", "L2"];

  const [batchSize, setBatchSize] = useState(0);
  const [learningRate, setLearningRate] = useState(0.03);
  const learningRates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
  const [regularisationRate, setRegularisationRate] = useState(0);
  const regularisationRates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

  const architectureParams = {
    networkShape: networkShape,
    activation: activation,
    outputActivation: activation,
    regularisation: regularisation,
    initZero: false
  }
  const trainingParams = {
    batchSize: batchSize,
    learningRate: learningRate,
    regLambda: regularisationRate,
    lossFn: "Square"
  }
  const buildParams = {"architectureParams": architectureParams, "trainingParams": trainingParams};

  var service: any;
  var label: number[] = [];
  const runParams = {"service": service, "label": label};

  const handleStart = () => {
    setStart(!start);
  };
  const handleReset = () => {
    setStart(false);
    setEpochs(0);
  };
  const handleStep = async () => {
    tick();
  };

  const tick = async () => {
    if (service == undefined) {
      try {
        const res = (await post("/classifier/classic/start", buildParams));
        service = res.service;
      } catch (error) {
        console.error('Error in API request', error);
      }
    } else {
      try {
        const res = (await post("/classifier/classic/run", runParams));
        setEpochs(res.epochs);
      } catch (error) {
        console.error('Error in API request', error);
      }
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (start) {
      interval = setInterval(async () => {
        tick();
      }, 50);
    }

    return () => {
      clearInterval(interval);
    };
  }, [start]);

  return (
    <div className="flex flex-col">
      {/* Top Controls */}
      <div className="flex flex-row px-40 items-center justify-between">
        {/* Control Menu */}
        <div className="flex flex-row h-full items-center gap-20">

          {/* Simulation Controls */}
          <div className="flex flex-row h-full items-center gap-3">
            <button onClick={handleReset}>
              <ArrowUturnLeftIcon className="w-7 h-7 p-1 flex-shrink-0" />
            </button>
            <button className="flex h-full aspect-square items-center justify-center rounded-full bg-aymlBlue bg-opacity-30" onClick={handleStart}>
              {start ? <PauseIcon className="w-8 h-8" /> : <PlayIcon className="w-8 h-8" />}
            </button>
            <button onClick={handleStep}>
              <ChevronDoubleRightIcon className="w-7 h-7 p-1 flex-shrink-0" />
            </button>
          </div>
          
          {/* Statistics */}
          <div className="flex flex-col h-full justify-between">
            <span className="text-xs">Epochs</span>
            <span className="text-2xl">{epochString.slice(0, 3) + ',' + epochString.slice(3)}</span>
          </div>
        </div>

        {/* Parameter Controls */}
        <div className="flex flex-row h-full items-center gap-8">
          {/* Learning Rates Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">Learning Rate</span>
            <Dropdown
              options={learningRates.map((r) => r.toString())}
              onChange={(selected) => setLearningRate(parseInt(selected))}
              defaultValue="0.03"
            />
          </div>

          {/* Activations Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">Activation</span>
            <Dropdown
              options={activations}
              onChange={(selected) => setActivation(selected)}
              defaultValue="Tanh"
            />
          </div>

          {/* Regularisation Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">Regularisation</span>
            <Dropdown
              options={regularisations}
              onChange={(selected) => setRegularisation(selected)}
              defaultValue="None"
            />
          </div>

          {/* Regularisation Rate Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">Regularisation Rate</span>
            <Dropdown
              options={regularisationRates.map((r) => r.toString())}
              onChange={(selected) => setRegularisationRate(parseInt(selected))}
              defaultValue="0"
            />
          </div>
        </div>
      </div>
      <hr className="my-8 mx-16 bg-black/20"/>

      {/* Bottom Controls */}
    </div>
  )
}