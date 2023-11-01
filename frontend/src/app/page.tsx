import Image from 'next/image'

export default function Home() {
  return (
    <div className="">
          <div className="relative mx-auto top-40 w-96 h-96 ">
            <div className="absolute rounded-full w-full h-full border border-gray-300 animate-orbit">
              <div className="absolute -top-4 left-36 rounded-full w-8 h-8 border border-green-800 animate-fastOrbit">
                <div className="relative top-8 left-16 rounded-full w-4 h-4 bg-green-800"/>
              </div>
            </div>
            <div className="absolute rounded-full w-full h-full border-2 border-green-800 border-t-transparent border-b-transparent animate-ellipsicalRotate"/>
          </div>
    </div>      
  )
}
