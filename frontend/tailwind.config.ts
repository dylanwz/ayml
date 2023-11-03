import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        transparent: 'transparent',
        current: 'currentColor',
        "aymlYellow": "#fca311",
        "aymlBlue": "#14213d",
        "aymlLightGray": "#e5e5e5",
      },
      keyframes: {
        ellipsicalRotate: {
          "0%": { transform: "rotate(-45deg)" },
          "100%": { transform: "rotate(-405deg)" },
        },
        orbit: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
      animation: {
        ellipsicalRotate: "ellipsicalRotate 15s ease-in-out infinite",
        orbit: "orbit 15s ease-in-out infinite",
        fastOrbit: "orbit 8s ease-in-out infinite",
      }
    },
  },
  plugins: [],
}
export default config
