/**
 * GeminiAdvisor.jsx
 * A live AI training advisor powered by Google Gemini 1.5 Flash.
 * Reads live training stats and answers RL questions with specific context.
 * Uses the Gemini REST API directly (no npm package needed).
 */
import { useState, useRef, useEffect } from 'react';

const GEMINI_MODEL = 'gemini-1.5-flash';
const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

const QUICK_PROMPTS = [
  { label: 'Analyze training', prompt: 'Analyze my current training metrics. Is the agent learning properly? What do you see?' },
  { label: 'Why is reward flat?', prompt: 'My average reward seems to be flatlining. What are the likely causes and what should I try?' },
  { label: 'Tune hyperparams', prompt: 'Based on my current stats, which hyperparameters should I adjust and in which direction?' },
  { label: 'Explain epsilon', prompt: 'Explain what my current epsilon value means for exploration vs exploitation and when I should expect it to drop.' },
  { label: 'Loss spike?', prompt: 'My loss had a spike. Is this normal? Should I be worried?' },
  { label: 'When does it learn?', prompt: 'At what point in training should I expect the agent to start visibly improving its gameplay?' },
];

function buildSystemPrompt(stats, algorithm) {
  return `You are an expert reinforcement learning engineer specializing in DQN variants for Atari games. 
You are advising someone training a Breakout agent in a custom training studio.

Current training state:
- Algorithm: ${algorithm ?? 'Unknown'}
- Episode: ${stats.episode ?? 0}
- Current reward (last ep): ${(stats.reward ?? 0).toFixed(1)}
- Average reward (last 100 eps): ${(stats.avgReward ?? 0).toFixed(2)}
- Loss (avg last 100): ${(stats.loss ?? 0).toFixed(4)}
- Epsilon (exploration rate): ${(stats.epsilon ?? 1).toFixed(4)}
- Mean Q-value: ${(stats.qValue ?? 0).toFixed(3)}
- Training speed: ${Math.round(stats.speed ?? 0)} episodes/hr

Environment details:
- State vector: 101 features (paddle pos, ball pos/vel, brick HP grid)
- Actions: 3 (left, right, noop)
- Reward shaping: +10 per brick, -5 per life lost, +50 per level, 100 levels total
- Replay buffer: 100K transitions, training starts after 1000 transitions
- Target network: synced every 10,000 steps

Give short, specific, actionable answers. Use plain English — avoid jargon unless asked. Be direct.`;
}

async function callGemini(apiKey, messages, stats, algorithm) {
  const systemPrompt = buildSystemPrompt(stats, algorithm);
  
  const contents = messages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.text }],
  }));

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents,
    generationConfig: {
      temperature: 0.7,
      maxOutputTokens: 512,
    },
  };

  const res = await fetch(
    `${GEMINI_API_BASE}/${GEMINI_MODEL}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error?.message ?? `HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '(no response)';
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function GeminiAdvisor({ stats, algorithm, isOpen, onToggle }) {
  const [apiKey, setApiKey]       = useState(() => localStorage.getItem('gemini_api_key') ?? '');
  const [showKey, setShowKey]     = useState(false);
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState('');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const messagesEndRef            = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const saveKey = (key) => {
    setApiKey(key);
    localStorage.setItem('gemini_api_key', key);
  };

  const sendMessage = async (text) => {
    if (!text.trim() || loading) return;
    if (!apiKey.trim()) { setError('Enter your Gemini API key first.'); return; }

    setError(null);
    const userMsg = { role: 'user', text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const reply = await callGemini(apiKey, newMessages, stats, algorithm);
      setMessages(prev => [...prev, { role: 'assistant', text: reply }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  const clearChat = () => { setMessages([]); setError(null); };

  return (
    <div style={{ ...styles.root, height: isOpen ? 520 : 44 }}>
      {/* Header / Toggle */}
      <button style={styles.header} onClick={onToggle} id="btn-gemini-toggle">
        <div style={styles.headerLeft}>
          <GeminiIcon />
          <span style={styles.headerTitle}>Gemini AI Advisor</span>
          {loading && <span style={styles.thinkingDot}>thinking...</span>}
        </div>
        <span style={styles.chevron}>{isOpen ? '▾' : '▸'}</span>
      </button>

      {isOpen && (
        <div style={styles.body}>
          {/* API key row */}
          <div style={styles.keyRow}>
            <div style={styles.keyInputWrap}>
              <input
                type={showKey ? 'text' : 'password'}
                placeholder="Paste Gemini API key (stored locally)"
                value={apiKey}
                onChange={e => saveKey(e.target.value)}
                className="input"
                style={styles.keyInput}
                id="gemini-api-key-input"
              />
              <button
                style={styles.showKeyBtn}
                onClick={() => setShowKey(s => !s)}
                title={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? '🙈' : '👁'}
              </button>
            </div>
            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noreferrer"
              style={styles.getKeyLink}
            >
              Get free key ↗
            </a>
          </div>

          {/* Quick prompts */}
          <div style={styles.quickRow}>
            {QUICK_PROMPTS.map(q => (
              <button
                key={q.label}
                className="btn btn-sm"
                style={styles.quickBtn}
                onClick={() => sendMessage(q.prompt)}
                disabled={loading || !apiKey}
                id={`gemini-quick-${q.label.replace(/\s+/g, '-').toLowerCase()}`}
              >
                {q.label}
              </button>
            ))}
          </div>

          {/* Messages */}
          <div style={styles.messages}>
            {messages.length === 0 && (
              <div style={styles.empty}>
                Ask Gemini anything about your training run — or tap a quick prompt above.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} style={m.role === 'user' ? styles.userBubble : styles.aiBubble}>
                <span style={m.role === 'user' ? styles.userLabel : styles.aiLabel}>
                  {m.role === 'user' ? 'You' : '✦ Gemini'}
                </span>
                <p style={styles.bubbleText}>{m.text}</p>
              </div>
            ))}
            {loading && (
              <div style={styles.aiBubble}>
                <span style={styles.aiLabel}>✦ Gemini</span>
                <p style={styles.bubbleText}><TypingDots /></p>
              </div>
            )}
            {error && (
              <div style={styles.errorMsg}>⚠ {error}</div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={styles.inputRow}>
            <textarea
              className="input"
              style={styles.textarea}
              placeholder="Ask about your training… (Enter to send)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={loading || !apiKey}
              id="gemini-chat-input"
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <button
                className="btn btn-primary btn-sm"
                style={{ whiteSpace: 'nowrap' }}
                onClick={() => sendMessage(input)}
                disabled={loading || !apiKey || !input.trim()}
                id="btn-gemini-send"
              >
                Send
              </button>
              <button
                className="btn btn-sm"
                style={{ fontSize: 10 }}
                onClick={clearChat}
                id="btn-gemini-clear"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return <span style={styles.dots}>● ● ●</span>;
}

function GeminiIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
      <path d="M12 2L9.5 9.5L2 12l7.5 2.5L12 22l2.5-7.5L22 12l-7.5-2.5L12 2z"
        fill="url(#gemini-grad)" />
      <defs>
        <linearGradient id="gemini-grad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop stopColor="#60a5fa" />
          <stop offset="1" stopColor="#a78bfa" />
        </linearGradient>
      </defs>
    </svg>
  );
}

const styles = {
  root: {
    background: 'var(--bg-800)',
    border: '1px solid rgba(96,165,250,0.25)',
    borderRadius: 10,
    overflow: 'hidden',
    display: 'flex', flexDirection: 'column',
    transition: 'height 0.25s ease',
    boxShadow: '0 0 20px rgba(96,165,250,0.08)',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 14px', height: 44, flexShrink: 0,
    background: 'transparent', border: 'none',
    cursor: 'pointer', width: '100%',
    borderBottom: '1px solid var(--border)',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 8 },
  headerTitle: { fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.02em' },
  thinkingDot: { fontSize: 10, color: '#60a5fa', fontStyle: 'italic' },
  chevron: { color: 'var(--text-muted)', fontSize: 14 },
  body: {
    flex: 1, display: 'flex', flexDirection: 'column', gap: 8,
    padding: 10, overflow: 'hidden',
  },
  keyRow: { display: 'flex', alignItems: 'center', gap: 8 },
  keyInputWrap: { flex: 1, position: 'relative' },
  keyInput: { width: '100%', paddingRight: 32, fontSize: 12 },
  showKeyBtn: {
    position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
    background: 'none', border: 'none', cursor: 'pointer', fontSize: 13,
  },
  getKeyLink: { fontSize: 11, color: '#60a5fa', whiteSpace: 'nowrap', textDecoration: 'none' },
  quickRow: { display: 'flex', flexWrap: 'wrap', gap: 5 },
  quickBtn: {
    fontSize: 10, padding: '3px 8px', background: 'transparent',
    borderColor: 'rgba(96,165,250,0.3)', color: '#93c5fd',
  },
  messages: {
    flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8,
    padding: '2px 0',
  },
  empty: {
    color: 'var(--text-muted)', fontSize: 12, textAlign: 'center',
    padding: '20px 12px', fontStyle: 'italic',
  },
  userBubble: {
    background: 'var(--bg-900)', borderRadius: 8, padding: '7px 10px',
    alignSelf: 'flex-end', maxWidth: '85%',
    border: '1px solid var(--border)',
  },
  aiBubble: {
    background: 'rgba(96,165,250,0.07)', borderRadius: 8, padding: '7px 10px',
    alignSelf: 'flex-start', maxWidth: '95%',
    border: '1px solid rgba(96,165,250,0.18)',
  },
  userLabel: { fontSize: 9, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: 3 },
  aiLabel: { fontSize: 9, fontWeight: 700, color: '#60a5fa', textTransform: 'uppercase', display: 'block', marginBottom: 3 },
  bubbleText: { margin: 0, fontSize: 12, lineHeight: 1.6, color: 'var(--text-primary)', whiteSpace: 'pre-wrap' },
  errorMsg: {
    color: 'var(--accent-red)', fontSize: 11, padding: '6px 10px',
    background: 'rgba(239,68,68,0.08)', borderRadius: 6,
    border: '1px solid rgba(239,68,68,0.2)',
  },
  inputRow: { display: 'flex', gap: 8, alignItems: 'flex-end' },
  textarea: { flex: 1, resize: 'none', fontSize: 12, lineHeight: 1.5, minHeight: 46 },
  dots: { color: '#60a5fa', letterSpacing: 3, animation: 'pulse 1.2s infinite' },
};
