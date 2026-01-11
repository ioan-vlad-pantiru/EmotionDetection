import { useState, useEffect } from 'react'
import EmotionDetector from './components/EmotionDetector'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold text-slate-800 mb-2">
            Emotion Detection
          </h1>
          <p className="text-slate-600 text-lg">
            Bilingual emotion detection using lexicon and TF-IDF features
          </p>
        </header>
        <EmotionDetector />
      </div>
    </div>
  )
}

export default App
