import Image from 'next/image'

export default function Home() {
  return (
    <div className="flex flex-col w-screen h-screen gap-24 justify-center items-center bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-aymlBlue to-slate-700">
      <span className="mt-20 text-white font-bold text-4xl text-center">Neural Networks<br></br>As You (Machine) Learn</span>
      <div className="flex justify-center items-center">
        
        {/* Neuron Element */}
        <div className="rounded-full w-80 h-80 border border-white animate-orbit">
          <div className="absolute -top-4 left-36 rounded-full w-8 h-8 border border-aymlYellow animate-fastOrbit">
            <div className="relative top-8 left-16 rounded-full w-4 h-4 bg-aymlYellow"/>
          </div>
        </div>
        <div className="absolute rounded-full w-80 h-80 border-2 border-aymlYellow border-t-transparent border-b-transparent animate-ellipsicalRotate"/>
        {/* Button */}
        <a href="/architecture" className="absolute py-2 px-4 rounded-full bg-slate-700 hover:bg-slate-800 text-white font-bold animate-bounce">
          <span>Select Architecture</span>
        </a>
      </div>
    </div>      
  )
}
