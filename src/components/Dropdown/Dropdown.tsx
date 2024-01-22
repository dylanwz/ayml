"use client";

import { useCallback, useEffect, useState } from "react";
import { Listbox } from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/24/solid";

export default function Dropdown({
  options,
  defaultValue,
  onChange,
  placeholder,
  manualChange
}: {
  options: string[];
  defaultValue?: string | null;
  onChange?: (selected: string) => void;
  placeholder?: string;
  manualChange?: boolean;
}) {
  const [selected, setSelected] = useState(defaultValue ?? "");

  useEffect(() => {
    if (!defaultValue) return;
    setSelected(defaultValue as string);
  }, [defaultValue]);

  // Handle dropdown value change
  const handleOnChange = useCallback(
    (value: string) => {
      if (onChange) {
        onChange(value);
      }
      if (manualChange === undefined) {
        setSelected(value);
      }
    },
    [onChange]
  );

  return (
    <Listbox value={selected} onChange={handleOnChange}>
      <div className="relative">
        {/* Dropdown when not expanded */}
        <Listbox.Button className="relative flex w-full py-2 items-center justify-between bg-white border border-t-transparent border-x-transparent border-slate-300 hover:border-b-aymlBlue aria-expanded:px-2 aria-expanded:rounded-md aria-expanded:border aria-expanded:border-b-transparent aria-expanded:rounded-b-none aria-expanded:border-aymlBlue">
          <span className="block truncate text-xs">
            {selected ? (
              <span>{selected}</span>
            ) : (
              <span>
                {placeholder}
              </span>
            )}
          </span>
          <span className="pointer-events-none">
            <ChevronDownIcon className="h-3 w-3" aria-hidden="true" />
          </span>
        </Listbox.Button>
        {/* Dropdown options */}
        <Listbox.Options className="absolute z-10 max-h-96 text-left w-full overflow-auto bg-white border border-aymlBlue rounded-md rounded-t-none">
          {options.map((option, index) => (
            <Listbox.Option
              key={index}
              className={({ active }) =>
                `relative cursor-pointer select-none p-2 ${
                  active ? "bg-aymlBlue/20" : "bg-white"
                }`
              }
              value={option}
            >
              {({ selected }) => (
                <span
                  className={`block truncate text-xs ${
                    selected ? "font-bold" : "font-normal"
                  }`}
                >
                  {option}
                </span>
              )}
            </Listbox.Option>
            ))}
        </Listbox.Options>
      </div>
    </Listbox>
  );
}