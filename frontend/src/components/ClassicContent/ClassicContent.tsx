'use client'

import { useEffect, useState } from "react";
import { ArrowUturnLeftIcon } from "@heroicons/react/24/outline";
import { PlayIcon } from "@heroicons/react/24/solid";
import { PauseIcon } from "@heroicons/react/24/solid";
import { ChevronDoubleRightIcon } from "@heroicons/react/24/outline";
import Dropdown from "@/components/Dropdown/Dropdown";
import Textform from "@/components/Textform/Textform";
import Image from "next/image";
import MNISTIcon from "@/assets/MNISTIcon.png";

import { IBuildRes, IRunRes } from "@/app/types/INetwork";
import { buildShape, startNetwork, runNetwork } from "@/app/utils/networkRequest";


export default function ClassicContent() {
  const [start, setStart] = useState(false);
  const [epochs, setEpochs] = useState(0);
  const [loss, setLoss] = useState(0.5);
  const [serviceID, setServiceID] = useState("-1")
  const epochString = ('000000' + epochs).slice(-6);

  const [dataspace, setDataspace] = useState("MNIST")
  const [networkShape, setNetworkShape] = useState([784, 10]);
  const [numHiddenLayers, setNumHiddenLayers] = useState(1);
  const [numHiddenNeurons, setNumHiddenNeurons] = useState(128);
  const [activation, setActivation] = useState("Tanh");
  const activations = ["ReLU", "Tanh", "Sigmoid", "Linear"];
  const [regularisation, setRegularisation] = useState("None");
  const regularisations = ["None", "L1", "L2"];

  const [batchSize, setBatchSize] = useState(10);
  const [learningRate, setLearningRate] = useState(0.03);
  const learningRates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
  const [regularisationRate, setRegularisationRate] = useState(0);
  const regularisationRates = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

  const architectureParams = {
    networkShape: buildShape(networkShape, numHiddenLayers, numHiddenNeurons),
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

  const handleStart = () => {
    setStart(!start);
  };
  const handleReset = () => {
    setStart(false);
    setServiceID("-1")
    setEpochs(0);
  };
  const handleStep = async () => {
    if (serviceID === "-1") {
      tick().then(() => tick());
    } else {
      tick();
    }
  };

  const onStartTick = (res: IBuildRes) => {
    setServiceID(res.serviceID);
    console.log(serviceID)
  }

  const onRunTick = (res: IRunRes) => {
    setEpochs(res.epochs);
    setLoss(res.loss);
  }

  const tick = async () => {
    if (serviceID === "-1") {
      try {
        startNetwork(architectureParams, trainingParams, onStartTick)
      } catch (error) {
        console.error('Error in API request', error);
      }
    } else {
      try {
        runNetwork(serviceID, dataspace, onRunTick);
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
              <ArrowUturnLeftIcon className="w-5 h-5 flex-shrink-0" />
            </button>
            <button className="flex h-full aspect-square items-center justify-center rounded-full bg-aymlBlue bg-opacity-30" onClick={handleStart}>
              {start ? <PauseIcon className="w-8 h-8" /> : <PlayIcon className="w-8 h-8" />}
            </button>
            <button onClick={handleStep}>
              <ChevronDoubleRightIcon className="w-5 h-5 flex-shrink-0" />
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
      <div className="flex flex-row px-40 gap-16">

        {/* Dataset Controls */}
        <div className="flex flex-col w-1/6 gap-3 items-start">
          <span>DATA</span>
          <span className="text-xs">Which dataset do you <br/> want to use?</span>
          {/* Dataset Options */}
          <div className="flex flex-row">
            <div className={`w-10 h-10 ${dataspace === "MNIST" ? "border border-black" : ""} hover:border hover:border-black`}>
              <Image
                src={MNISTIcon}
                width={40}
                height={40}
                alt={"MNIST Icon"}
              />
            </div>
          </div>
          <div className="flex flex-col w-full text-xs text-gray-400">
            <div className="flex flex-row justify-between">
              <span>Input Layer:</span>
              <span>{networkShape[0]}</span>
            </div>
            <div className="flex flex-row justify-between">
              <span>Output Layer:</span>
              <span>{networkShape[networkShape.length - 1]}</span>
            </div>
          </div>
          {/* Batch Size Options */}
          <div className="flex flex-row justify-between w-full text-xs mt-3">
            <span>Batch Size:</span>
            <Textform 
              defaultValue={"10"}
              passCondition={(s: string) => /^\d*$/.test(s)}
              inputModifier={(s: string) => s.length <= 2 ? s : s.slice(0, 2)}
              onChange={(input) => setBatchSize(parseInt(input))}
            />
          </div>
        </div>

        {/* Training Controls */}
        <div className="flex flex-col w-2/3 items-center gap-3">
          <span>TRAINING</span>
          <div className="flex flex-col w-1/2 items-center gap-2">
            <div className="flex flex-row justify-between text-xs">
              <span>Hidden Layers:</span>
              <Textform 
                defaultValue={"1"}
                passCondition={(s: string) => /^\d*$/.test(s)}
                inputModifier={(s: string) => parseInt(s) > 100 ? "100" : s}
                onChange={(input) => setNumHiddenLayers(parseInt(input))}
              />
            </div>
            <div className="flex flex-row justify-between text-xs">
              <span>Neurons:</span>
              <Textform 
                defaultValue={"128"}
                passCondition={(s: string) => /^\d*$/.test(s)}
                inputModifier={(s: string) => parseInt(s) > 100 ? "100" : s}
                onChange={(input) => setNumHiddenNeurons(parseInt(input))}
              />
            </div>
          </div>
        </div>

        {/* Results Controls */}
        <div className="flex flex-col w-1/6 items-end gap-3">
          <span>RESULTS</span>
          <div className="flex flex-col w-full text-xs items-end">
            <div className="flex flex-row w-full justify-between">
              <span>Test Loss:</span>
              <span>{loss.toFixed(3)}</span>
            </div>
            <div className="flex flex-row w-full justify-between text-gray-400">
              <span>Training Loss:</span>
              <span>{loss.toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}