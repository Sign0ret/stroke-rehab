import { Button } from "@/components/ui/button";
import { TracingBeam } from "@/components/ui/tracing-beam";
import Link from "next/link";
import { Members } from "@/components/members";

export default function Home() {
  return (
    <TracingBeam className="px-6">
      <main className="flex min-h-screen flex-col items-center justify-left p-24 bg-black text-white">
        {/* Project Title */}
        <section className="mb-10">
          <div className="flex flex-row gap-4 items-center mb-4">
            <h1 className="text-4xl font-bold">The Project</h1>
            <Link href="/motor">
              <Button className="hover:border-white">
                Wanna try it !?
              </Button>
            </Link>
          </div>

          <p className="text-lg">
            Roberto Morales, Adolfo Hernández, José Emilio Inzunza, Esteban Ochoa, Esteban Muñoz, Alonso Rivera and Isabella Hurtado
          </p>
        </section>
        
        {/* Initial Situation Section */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-2">Initial Situation</h2>
          <p className="text-lg leading-relaxed max-w-2xl text-left">
            The project aims to analyze a motor imagery dataset from a chronic stroke patient to optimize pre-processing,
            feature extraction, and classification algorithms for detecting arm movements. The challenge lies in designing
            a robust pipeline that enhances the quality of EEG signals through advanced techniques such as band-specific
            filtering and Common Spatial Patterns (CSP).
          </p>
        </section>
        
        {/* Idea and Solution */}
        <section className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Idea and Solution</h1>
          <h2 className="text-3xl font-semibold mb-2">The Idea</h2>
          <p className="text-lg leading-relaxed max-w-2xl text-left">
            Our approach is to develop a streamlined processing pipeline that can be adapted across different patient datasets.
          </p>
          <h2 className="text-3xl font-semibold mb-2">The Solution</h2>
          <p className="text-lg leading-relaxed max-w-2xl text-left">
            We implemented band-pass filtering, spatial feature extraction using CSP, and optimized classifiers for improved performance.
          </p>
        </section>
        
        {/* Implementation Realization */}
        <section className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Our Implementationn</h1>
          <p className="text-lg leading-relaxed max-w-2xl text-left">
          </p>
        </section>
        
        {/* Results Outcome */}
        <section className="mb-10" >
          <h1 className="text-4xl font-bold mb-4">Results</h1>
        </section>
        
        {/* Reflection Section */}
        <section className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Reflection</h1>
          <p className="text-lg leading-relaxed max-w-2xl text-left">
          This project allowed us to delve into the complexities of EEG signal 
          analysis in chronic stroke patients. Implementing advanced techniques
          for filtering, feature extraction, and classification highlighted the
          importance of adapting methods to patient variability. The outcomes 
          demonstrate the potential of these approaches to improve motor imagery
          detection, contributing to more effective neurorehabilitation strategies.
          </p>
        </section>
        
        {/* Group Picture (pics working) */}
        <section className="mb-10">
          <h1 className="text-4xl font-bold mb-4">The Team</h1>
          <img src="/simbolo-tec-black.png" alt="Team Photo" width="400" className="mb-4" />
        </section>
        
        {/* Group Information Section */}
        <section className="mb-10 text-center">
          <h1 className="text-4xl font-bold mb-4">Stroke Rehab Data Analysis</h1>
          <Members/>
        </section>
        
      </main>
    </TracingBeam>
    
  );
}
