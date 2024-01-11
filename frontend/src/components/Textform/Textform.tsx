"use  client"

import { useRef, useState } from "react";

export default function Textform({
  defaultValue,
  passCondition,
  inputModifier,
  onChange,
  disabled
}: {
  defaultValue?: string | null;
  passCondition?: (inpupt: string) => boolean;
  inputModifier?: (input: string) => string;
  onChange: (input: string) => void;
  disabled?: boolean
}) {

  const [input, setInput] = useState(defaultValue ?? "");

  const handleWrite = (s: string) => {
    const tautCondition = (s: string) => {return true};
    const tautModifier = (s: string) => {return s};
    const condition = passCondition ?? tautCondition;
    const modifier = inputModifier ?? tautModifier
    if (condition(s)) {
      setInput(modifier(s));
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onChange(input);
    }
  };

  return (
    <div className="flex items-center">
      <input
        type="text"
        className="ml-2 w-8 border-b border-aymlBlue text-center text-xs"
        value={input}
        onChange={e => handleWrite(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled ?? false}
      />
    </div>
  );
}