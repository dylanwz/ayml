'use client'

import { useEffect, useState } from "react";
import { ArrowUturnLeftIcon } from "@heroicons/react/24/outline";
import { PlayIcon } from "@heroicons/react/24/solid";
import { PauseIcon } from "@heroicons/react/24/solid";
import { ChevronDoubleRightIcon } from "@heroicons/react/24/outline";
import { PlusIcon } from "@heroicons/react/24/solid";
import { MinusIcon } from "@heroicons/react/24/solid";
import Dropdown from "@/components/Dropdown/Dropdown";
import Textform from "@/components/Textform/Textform";
import Image from "next/image";
import MNISTIcon from "@/assets/MNISTIcon.png";

import { IBuildRes, IRunRes } from "@/app/types/INetwork";

export default function HomeContent() {
  
  // Control States
  const [start, setStart] = useState(false);
  const [intiatied, setInitiated] = useState(false);

  // Datasets
  const datasets = ["MNIST"];
  const [dataset, setDataset] = useState(datasets[0]);

  // Shape
  const MAXHIDDENLAYERS = 2;
  const MAXCONVLAYERS = 4;
  const MINLAYERS = 0;
  const [convShape, setConvShape] = useState([[8,8]]);
  const [hiddenShape, setHiddenShape] = useState([1,1]);

  // Batch Sizes
  const [batchSize, setBatchSize] = useState(10);
  
  // Learning Rates
  const learningRates = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
  const [convLearningRate, setConvLearningRate] = useState(learningRates[3]);
  const [hiddenLearningRate, setHiddenLearningRate] = useState(learningRates[3]);

  // Activations
  const activations = ["ReLU", "Tanh", "Sigmoid", "Linear"];
  const [convActivation, setConvActivation] = useState(activations[0]);
  const [hiddenActivation, setHiddenActivation] = useState(activations[0]);

  // Statistics
  const [loss, setLoss] = useState(0);
  const [epochs, setEpochs] = useState(0);
  let epochAsInt = ('000000' + epochs).slice(-6);
  let epochString = epochAsInt.slice(0,3) + ',' + epochAsInt.slice(3);

  // Top Control Functions
  const handleStart = () => {
    setStart(!start);
  };
  const handleReset = () => {
    setStart(false);
    setInitiated(false);
    setEpochs(0);
  };
  const handleStep = async () => {
    tick();
  };

  // Frontend-Network API Functions
  function onStartTick(res: IBuildRes) {
    setInitiated(true);
  }
  function onRunTick(res: IRunRes) {
    setEpochs(res.epochs);
    setLoss(res.loss);
  }
  function tick() {
    return;
  }
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (start) {
      interval = setInterval(() => {
        tick();
      }, 50);
    }
    return () => {
      clearInterval(interval);
    };
  }, [start]);

  // Build Network Functions
  const onAddConvLayer = () => {
    if (convShape.length < MAXCONVLAYERS) {
      const newShape = [...convShape];
      newShape.push([8,8]);
      setConvShape(newShape);
    }
  };
  const onRemoveConvLayer = () => {
    if (convShape.length > MINLAYERS) {
      const newShape = [...convShape];
      newShape.pop();
      setConvShape(newShape);
    }
  };
  const onAddHiddenLayer = () => {
    if (hiddenShape.length < MAXHIDDENLAYERS) {
      const newShape = [...hiddenShape];
      newShape.push(2);
      setHiddenShape(newShape);
    }
  };
  const onRemoveHiddenLayer = () => {
    if (hiddenShape.length > MINLAYERS) {
      const newShape = [...hiddenShape];
      newShape.pop();
      setHiddenShape(newShape);
    }
  };

  return (
    <div className="flex flex-col py-20 px-10 w-2/3">

      {/* Top Controls */}
      <div className="flex flex-row items-center justify-between">

        {/* Simulation Features */}
        <div className="flex flex-row w-64 items-center justify-between">

          {/* Timeline Controls */}
          <div className="flex flex-row gap-4">
            <button onClick={handleReset}>
              <ArrowUturnLeftIcon className="w-5 h-5 flex-shrink-0" />
            </button>
            <button className="flex h-full aspect-square items-center justify-center rounded-full bg-aymlBlue bg-opacity-30 hover:bg-opacity-40" onClick={handleStart}>
              {start ? <PauseIcon className="w-14 h-14 p-2" /> : <PlayIcon className="w-14 h-14 p-2" />}
            </button>
            <button onClick={handleStep}>
              <ChevronDoubleRightIcon className="w-5 h-5 flex-shrink-0" />
            </button>
          </div>

          {/* Statistics */}
          <div className="flex flex-col h-full justify-between">
            <span className="text-xs">Epochs</span>
            <span className="text-2xl">{epochString}</span>
          </div>

        </div>

        {/* Parameter Controls */}
        <div className="flex flex-row w-[768] h-full items-center gap-8 justify-between">
          {/* Convolutional Learning Rates Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">(C) Learning Rate</span>
            <Dropdown
              options={learningRates.map((r) => r.toString())}
              onChange={(selected) => setConvLearningRate(parseInt(selected))}
              defaultValue="0.03"
            />
          </div>
          {/* Convolutional Activations Menu */}
          <div className="flex flex-col h-full w-28 justify-between gap-2">
            <span className="text-xs">(C) Activation</span>
            <Dropdown
              options={activations}
              onChange={(selected) => setConvActivation(selected)}
              defaultValue="ReLU"
            />
          </div>
          {/* Hidden Learning Rates Menu */}
          <div className="flex flex-col h-full w-32 justify-between gap-2">
            <span className="text-xs">(H) Learning Rate</span>
            <Dropdown
              options={learningRates.map((r) => r.toString())}
              onChange={(selected) => setHiddenLearningRate(parseInt(selected))}
              defaultValue="0.03"
            />
          </div>
          {/* Hidden Activations Menu */}
          <div className="flex flex-col h-full w-28 justify-between gap-2">
            <span className="text-xs">(H) Activation</span>
            <Dropdown
              options={activations}
              onChange={(selected) => setHiddenActivation(selected)}
              defaultValue="Tanh"
            />
          </div>

        </div>
      </div>

      <hr className="my-8 bg-black/20"/>

      {/* Bottom Controls */}
      <div className="flex flex-row gap-16">

        {/* Dataset Controls */}
        <div className="flex flex-col w-1/6 gap-3 items-start">
          <span>DATA</span>
          <span className="text-xs">Which dataset do you <br/> want to use?</span>
          {/* Dataset Options */}
          <div className="flex flex-row">
            <div className={`w-10 h-10 hover:border hover:border-black ${dataset === "MNIST" ? "border border-black" : ""}`}>
              <Image
                src={MNISTIcon}
                width={40}
                height={40}
                alt={"MNIST Icon"}
              />
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
        <div className="flex flex-row w-2/3 justify-between gap-3">
          
          {/* Convolutional Shape */}
          <div className="flex flex-col items-center w-2/3">
            <div className="flex flex-row justify-center items-center w-full gap-2">
                <button className="rounded-full bg-aymlBlue bg-opacity-10 hover:bg-opacity-20 w-6 h-6 p-1" onClick={onAddConvLayer}>
                  <PlusIcon  />
                </button>
                <button className="rounded-full bg-aymlBlue bg-opacity-10 hover:bg-opacity-20 w-6 h-6 p-1" onClick={onRemoveConvLayer}>
                  <MinusIcon />
                </button>
              <span className="pl-2">{convShape.length} CONVOLUTIONAL LAYERS</span>
            </div>
          </div>

          {/* Hidden Shape */}
          <div className="flex flex-col items-center w-1/3">
            <div className="flex flex-row justify-center items-center w-full gap-2">
                <button className="rounded-full bg-aymlBlue bg-opacity-10 hover:bg-opacity-20 w-6 h-6 p-1" onClick={onAddHiddenLayer}>
                  <PlusIcon  />
                </button>
                <button className="rounded-full bg-aymlBlue bg-opacity-10 hover:bg-opacity-20 w-6 h-6 p-1" onClick={onRemoveHiddenLayer}>
                  <MinusIcon />
                </button>
              <span className="pl-2">{hiddenShape.length} HIDDEN LAYERS</span>
            </div>
          </div>
        </div>

        {/* Results Controls */}
        <div className="flex flex-col w-1/6 items-end gap-3">
          <span>OUTPUT</span>
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