import { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || ''

function Badge({ label }) {
  const urgent = label === 'urgent'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 12, fontWeight: 700, letterSpacing: 1,
      background: urgent ? '#fee2e2' : '#d1fae5',
      color: urgent ? '#b91c1c' : '#065f46',
    }}>
      {label.toUpperCase()}
    </span>
  )
}

function Meter({ value }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 8, background: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: pct > 75 ? '#10b981' : pct > 50 ? '#f59e0b' : '#ef4444',
          borderRadius: 4, transition: 'width 0.4s',
        }} />
      </div>
      <span style={{ fontSize: 13, color: '#6b7280', minWidth: 36 }}>{pct}%</span>
    </div>
  )
}

function Card({ title, accent, children }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: '20px 24px',
      boxShadow: '0 1px 4px rgba(0,0,0,0.08)', borderTop: `4px solid ${accent}`,
    }}>
      <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#374151' }}>{title}</h3>
      {children}
    </div>
  )
}

function StatRow({ label, value, dim }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
      <span style={{ color: '#6b7280' }}>{label}</span>
      <span style={{ fontWeight: 600, color: dim ? '#9ca3af' : '#111827' }}>{value}</span>
    </div>
  )
}

function PriorityCard({ title, accent, data }) {
  if (!data) return null
  return (
    <Card title={title} accent={accent}>
      <div style={{ marginBottom: 10 }}><Badge label={data.label} /></div>
      <Meter value={data.confidence} />
      <div style={{ marginTop: 10 }}>
        <StatRow label="Confidence" value={`${Math.round((data.confidence ?? 0) * 100)}%`} />
        <StatRow label="Latency" value={data.latency_ms != null ? `${data.latency_ms} ms` : '—'} />
        <StatRow label="Cost" value={data.cost_usd > 0 ? `$${data.cost_usd.toFixed(7)}` : '$0.000 (free)'} dim={!data.cost_usd} />
        <StatRow label="Model" value={data.model_name ?? '—'} />
      </div>
      {data.reasoning && (
        <p style={{ marginTop: 10, fontSize: 13, color: '#4b5563', borderTop: '1px solid #f3f4f6', paddingTop: 10 }}>
          {data.reasoning}
        </p>
      )}
    </Card>
  )
}

function AnswerCard({ title, accent, data }) {
  if (!data) return null
  return (
    <Card title={title} accent={accent}>
      <p style={{ fontSize: 14, color: '#1f2937', lineHeight: 1.7, margin: '0 0 12px' }}>{data.answer}</p>
      <StatRow label="Latency" value={`${data.latency_ms} ms`} />
      <StatRow label="Cost" value={`$${data.cost_usd.toFixed(7)}`} />
      <StatRow label="Model" value={data.model_name} />
    </Card>
  )
}

function RetrievedCase({ c, index }) {
  const [open, setOpen] = useState(false)
  const pct = Math.round(c.similarity * 100)
  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, marginBottom: 8, overflow: 'hidden' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', background: '#f9fafb', border: 'none', cursor: 'pointer',
          padding: '10px 14px', display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', fontSize: 13,
        }}
      >
        <span style={{ color: '#374151' }}>Case {index + 1} — <Badge label={c.priority_label_str} /></span>
        <span style={{ color: '#6b7280' }}>Similarity: {pct}%　{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.6 }}>
          <p style={{ margin: '0 0 6px', color: '#6b7280' }}><strong>Customer:</strong> {c.question}</p>
          <p style={{ margin: 0, color: '#374151' }}><strong>Agent reply:</strong> {c.answer}</p>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post(`${API}/compare`, { text: text.trim() })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message ?? 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f3f4f6', padding: '32px 16px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 960, margin: '0 auto' }}>

        <div style={{ marginBottom: 32 }}>
          <h1 style={{ margin: '0 0 4px', fontSize: 26, color: '#111827' }}>Decision Intelligence Assistant</h1>
          <p style={{ margin: 0, color: '#6b7280', fontSize: 14 }}>RAG · LLM · ML Baseline · Side-by-Side Comparison</p>
        </div>

        <form onSubmit={handleSubmit} style={{ marginBottom: 32 }}>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste a customer support ticket here…"
            rows={4}
            style={{
              width: '100%', padding: '14px 16px', fontSize: 14, borderRadius: 10,
              border: '1px solid #d1d5db', resize: 'vertical', lineHeight: 1.6,
              background: '#fff', boxSizing: 'border-box', outline: 'none', fontFamily: 'inherit',
            }}
          />
          <div style={{ marginTop: 10, display: 'flex', gap: 10 }}>
            <button
              type="submit"
              disabled={loading || !text.trim()}
              style={{
                padding: '10px 28px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: loading || !text.trim() ? '#9ca3af' : '#2563eb',
                color: '#fff', fontWeight: 600, fontSize: 14,
              }}
            >
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
            {result && (
              <button
                type="button"
                onClick={() => { setResult(null); setText('') }}
                style={{
                  padding: '10px 18px', borderRadius: 8, border: '1px solid #d1d5db',
                  background: '#fff', cursor: 'pointer', fontSize: 14, color: '#374151',
                }}
              >
                Clear
              </button>
            )}
          </div>
        </form>

        {error && (
          <div style={{
            background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 8,
            padding: '12px 16px', color: '#b91c1c', marginBottom: 24, fontSize: 14,
          }}>
            {error}
          </div>
        )}

        {result && (
          <>
            <h2 style={{ fontSize: 16, color: '#374151', margin: '0 0 14px' }}>Priority Prediction</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
              <PriorityCard title="ML Baseline (Random Forest)" accent="#8b5cf6" data={result.ml_prediction} />
              <PriorityCard title="LLM Zero-Shot (GPT-4o-mini)" accent="#2563eb" data={result.llm_prediction} />
            </div>

            <div style={{
              background: '#fff', borderRadius: 12, padding: '16px 20px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 28,
            }}>
              <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#374151' }}>Priority Predictors — Head-to-Head</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #f3f4f6' }}>
                    {['', 'Label', 'Confidence', 'Latency', 'Cost'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '6px 10px', color: '#6b7280', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>ML (RF)</td>
                    <td style={{ padding: '8px 10px' }}><Badge label={result.ml_prediction.label} /></td>
                    <td style={{ padding: '8px 10px' }}>{Math.round((result.ml_prediction.confidence ?? 0) * 100)}%</td>
                    <td style={{ padding: '8px 10px' }}>{result.ml_prediction.latency_ms} ms</td>
                    <td style={{ padding: '8px 10px', color: '#10b981', fontWeight: 600 }}>$0.000 (free)</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>LLM Zero-Shot</td>
                    <td style={{ padding: '8px 10px' }}><Badge label={result.llm_prediction.label} /></td>
                    <td style={{ padding: '8px 10px' }}>{Math.round((result.llm_prediction.confidence ?? 0) * 100)}%</td>
                    <td style={{ padding: '8px 10px' }}>{result.llm_prediction.latency_ms} ms</td>
                    <td style={{ padding: '8px 10px' }}>${result.llm_prediction.cost_usd?.toFixed(7)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h2 style={{ fontSize: 16, color: '#374151', margin: '0 0 14px' }}>Generated Answer</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 28 }}>
              <AnswerCard title="Non-RAG Answer (LLM alone)" accent="#f59e0b" data={result.non_rag_answer} />
              <AnswerCard title={`RAG Answer (LLM + ${result.retrieved_cases?.length ?? 0} cases)`} accent="#10b981" data={result.rag_answer} />
            </div>

            {result.retrieved_cases?.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <h2 style={{ fontSize: 16, color: '#374151', margin: '0 0 14px' }}>
                  Retrieved Cases
                  <span style={{ fontSize: 12, color: '#9ca3af', fontWeight: 400, marginLeft: 10 }}>
                    retrieval: {result.retrieval_latency_ms} ms
                  </span>
                </h2>
                {result.retrieved_cases.map((c, i) => (
                  <RetrievedCase key={c.tweet_id} c={c} index={i} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
