import Image from "next/image";
import diver from "../assets/diver.svg";

export default function Home() {
  return (
    <div className="w-screen h-screen bg-black bg-center bg-contain bg-no-repeat bg-[url('../assets/land.png')]">

      {/* Navbar */}
      <div> 
      
      </div>

      {/* Hero Container */}
      <div className="flex flex-col h-full py-24 items-center justify-between">

        {/* Line */}
        <div className="inline-flex items-center justify-center w-3/4">
          <hr className="w-full h-[1px] my-8 bg-opacity-0 bg-slate-900 rounded"/>
        </div>
        
        {/* Hero Text */}
        <div className="flex items-center justify-center h-full w-full">
          <span className="text-9xl text-white">a y m l</span>
          <div className="absolute transform translate-x-8 rotate-180">
            <Image 
              src={diver}
              width={300}
              height={300}
              alt={"landing page diver"}
            />
          </div>
        </div>

        {/* Hero Button */}
        <div className="inline-flex items-center justify-between w-3/4 gap-16">
            <div className="w-1/3">
              <hr className="w-full h-[1px] my-8 bg-opacity-0 bg-slate-900 rounded"/>
            </div>
            <a href="/build" className="p-4 bg-white hover:bg-slate-300 rounded-tl-3xl rounded-br-3xl text-center">
                <span className="text-black text-sm">EXPLORE<br/>ARCHITECTURES</span>
            </a>
            <div className="w-1/3">
              <hr className="w-full h-[1px] my-8 bg-opacity-0 bg-slate-900 rounded"/>
            </div>
        </div>

      </div>
    </div>     
  );
}
