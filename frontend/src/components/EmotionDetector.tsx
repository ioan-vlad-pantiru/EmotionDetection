import { useState } from 'react'

interface PredictionResponse {
  emotion: string
  text: string
  lang: string
  model: string
  confidence: number
  probabilities: Record<string, number>
}

const EMOTION_COLORS: Record<string, string> = {
  anger: 'bg-red-500',
  fear: 'bg-purple-500',
  anticipation: 'bg-amber-500',
  trust: 'bg-blue-500',
  surprise: 'bg-pink-500',
  sadness: 'bg-cyan-500',
  joy: 'bg-green-500',
  disgust: 'bg-lime-500',
  neutral: 'bg-gray-500',
}

const EMOTION_LABELS: Record<string, string> = {
  anger: 'Anger',
  fear: 'Fear',
  anticipation: 'Anticipation',
  trust: 'Trust',
  surprise: 'Surprise',
  sadness: 'Sadness',
  joy: 'Joy',
  disgust: 'Disgust',
  neutral: 'Neutral',
}

export default function EmotionDetector() {
  const [text, setText] = useState('')
  const [lang, setLang] = useState('en')
  const [model, setModel] = useState('hybrid')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text.trim(),
          lang,
          model,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Prediction failed')
      }

      const data: PredictionResponse = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const emotionColor = result ? EMOTION_COLORS[result.emotion] || 'bg-gray-500' : ''
  const emotionLabel = result ? EMOTION_LABELS[result.emotion] || result.emotion : ''

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Language
            </label>
            <div className="flex gap-4">
              <button
                onClick={() => setLang('en')}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  lang === 'en'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                English
              </button>
              <button
                onClick={() => setLang('ro')}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  lang === 'ro'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                Romanian
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Model Type
            </label>
            <div className="flex gap-4 flex-wrap">
              {['lexicon', 'ml', 'hybrid'].map((modelType) => (
                <button
                  key={modelType}
                  onClick={() => setModel(modelType)}
                  className={`px-6 py-2 rounded-lg font-medium transition-colors capitalize ${
                    model === modelType
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {modelType === 'ml' ? 'ML (TF-IDF)' : modelType}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Text to Analyze
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to detect emotion..."
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={6}
            />
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || !text.trim()}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Analyzing...' : 'Detect Emotion'}
          </button>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}

          {result && (
            <div className="mt-6 space-y-4">
              <div className="p-6 bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl border border-slate-200">
                <div className="text-center">
                  <div className="inline-block mb-4">
                    <div
                      className={`${emotionColor} text-white px-8 py-4 rounded-xl text-2xl font-bold shadow-lg`}
                    >
                      {emotionLabel}
                    </div>
                  </div>
                  <div className="text-slate-600 text-sm mb-2">
                    Confidence: <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-slate-600 text-sm">
                    Detected using <span className="font-semibold capitalize">{result.model}</span> model
                    {' '}for <span className="font-semibold uppercase">{result.lang}</span>
                  </div>
                </div>
              </div>

              {text.trim().split(/\s+/).length < 3 && (
                <div className="bg-amber-50 border border-amber-200 text-amber-700 px-4 py-3 rounded-lg text-sm">
                  ⚠️ Very short text may produce less accurate results. Try using longer sentences for better predictions.
                </div>
              )}

              {result.confidence < 0.5 && (
                <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-lg text-sm">
                  ⚠️ Low confidence score. The prediction may be uncertain.
                </div>
              )}

              <div className="p-4 bg-white rounded-lg border border-slate-200">
                <h3 className="text-sm font-semibold text-slate-700 mb-3">All Emotion Probabilities:</h3>
                <div className="space-y-2">
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emotion, prob]) => (
                      <div key={emotion} className="flex items-center gap-3">
                        <div className="w-24 text-sm text-slate-600 capitalize">
                          {EMOTION_LABELS[emotion] || emotion}
                        </div>
                        <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
                          <div
                            className={`h-full ${EMOTION_COLORS[emotion] || 'bg-gray-400'} transition-all duration-300`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <div className="w-12 text-sm text-slate-600 text-right">
                          {(prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
