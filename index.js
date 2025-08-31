// index.js — Gemini 2.5 TTS server (Railway-friendly URLs)
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

// ---------- Config ----------
const PORT = process.env.PORT || 3000;
const DEFAULT_MODEL = process.env.MODEL || 'gemini-2.5-flash-preview-tts';
const DEFAULT_VOICE = process.env.VOICE || 'Kore';
const API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
if (!API_KEY) {
  console.error('❌ Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env');
  process.exit(1);
}

// ---------- SDK ----------
const ai = new GoogleGenAI({ apiKey: API_KEY });

// ---------- App ----------
const app = express();
app.set('trust proxy', true); // <-- important on Railway so we can read X-Forwarded-* headers
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// Static file hosting for URL-returned audio
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_AUDIO_DIR = path.join(__dirname, 'public', 'audio');
fs.mkdirSync(PUBLIC_AUDIO_DIR, { recursive: true });
app.use('/audio', express.static(PUBLIC_AUDIO_DIR)); // e.g., https://YOUR.up.railway.app/audio/<file>.wav

// ---------- Helpers ----------
function pcm16ToWav(pcmBuffer, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmBuffer.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmBuffer.length, 40);
  return Buffer.concat([header, pcmBuffer]);
}

// Build the public origin behind a proxy (Railway sets these headers)
function getPublicOrigin(req) {
  const proto = req.get('x-forwarded-proto') || req.protocol; // should be "https"
  const host = req.get('x-forwarded-host') || req.get('host'); // YOUR.up.railway.app
  return `${proto}://${host}`;
}

// Simple in-memory cache (cuts repeat latency)
const audioCache = new Map(); // key -> { wavBuf: Buffer, createdAt: number }
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const makeKey = ({ text, voice, model }) => `${model}::${voice}::${text}`;

async function synthesizeToWav({ text, voice, model }) {
  const key = makeKey({ text, voice, model });
  const now = Date.now();
  const hit = audioCache.get(key);
  if (hit && (now - hit.createdAt) < CACHE_TTL_MS) return hit.wavBuf;

  const response = await ai.models.generateContent({
    model,
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
    },
  });

  const base64 = response?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64) throw new Error('No audio produced by model');

  const pcm = Buffer.from(base64, 'base64');      // 24 kHz, 16-bit PCM mono
  const wavBuf = pcm16ToWav(pcm, 24000, 1, 16);   // wrap as WAV
  audioCache.set(key, { wavBuf, createdAt: now });
  return wavBuf;
}

// ---------- Routes ----------
app.get('/health', (_req, res) => {
  res.json({ ok: true, uptime: process.uptime(), model: DEFAULT_MODEL, voice: DEFAULT_VOICE });
});

app.get('/', (_req, res) => {
  res.send('Gemini TTS server ready. Endpoints: POST /tts, POST /tts-url, POST /batch-url, GET /health');
});

// POST /tts  (streams WAV or returns base64 JSON)
app.post('/tts', async (req, res) => {
  try {
    const { text, voice, model, return: ret } = req.body || {};
    const t = (text || '').toString();
    if (!t.trim()) return res.status(400).json({ error: 'Missing required "text" string.' });
    const useModel = model || DEFAULT_MODEL;
    const useVoice = voice || DEFAULT_VOICE;

    const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });

    if (ret === 'base64') {
      return res.json({
        audio: wavBuf.toString('base64'),
        mimeType: 'audio/wav',
        sampleRate: 24000,
        channels: 1,
        model: useModel,
        voice: useVoice
      });
    }
    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Disposition', 'inline; filename="speech.wav"');
    res.send(wavBuf);
  } catch (err) {
    console.error('POST /tts error:', err);
    res.status(500).json({ error: 'TTS failed', detail: err?.message });
  }
});

// POST /tts-url  (saves file and returns a public URL)
app.post('/tts-url', async (req, res) => {
  try {
    const { text, voice, model } = req.body || {};
    const t = (text || '').toString();
    if (!t.trim()) return res.status(400).json({ error: 'Missing required "text" string.' });
    const useModel = model || DEFAULT_MODEL;
    const useVoice = voice || DEFAULT_VOICE;

    const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });
    const id = crypto.randomUUID();
    const filename = `${id}.wav`;
    fs.writeFileSync(path.join(PUBLIC_AUDIO_DIR, filename), wavBuf);

    const url = `${getPublicOrigin(req)}/audio/${filename}`;
    res.json({ url, sampleRate: 24000, channels: 1, model: useModel, voice: useVoice });
  } catch (err) {
    console.error('POST /tts-url error:', err);
    res.status(500).json({ error: 'TTS failed', detail: err?.message });
  }
});

// POST /batch-url  (many clips → many URLs)
app.post('/batch-url', async (req, res) => {
  try {
    const { items } = req.body || {};
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: 'Provide items: an array of { text, voice?, model? }' });
    }

    const results = await Promise.all(items.map(async (item) => {
      try {
        const t = (item?.text || '').toString();
        if (!t.trim()) return { error: 'Missing text' };
        const useModel = item?.model || DEFAULT_MODEL;
        const useVoice = item?.voice || DEFAULT_VOICE;

        const wavBuf = await synthesizeToWav({ text: t, voice: useVoice, model: useModel });
        const id = crypto.randomUUID();
        const filename = `${id}.wav`;
        fs.writeFileSync(path.join(PUBLIC_AUDIO_DIR, filename), wavBuf);
        const url = `${getPublicOrigin(req)}/audio/${filename}`;
        return { text: t, url, model: useModel, voice: useVoice, sampleRate: 24000, channels: 1 };
      } catch (e) {
        return { error: e?.message || 'synthesis failed' };
      }
    }));

    res.json({ results });
  } catch (err) {
    console.error('POST /batch-url error:', err);
    res.status(500).json({ error: 'Batch TTS failed', detail: err?.message });
  }
});

// ---------- Start ----------
app.listen(PORT, () => {
  console.log(`✅ TTS server listening on http://localhost:${PORT}`);
  console.log('Routes: POST /tts, POST /tts-url, POST /batch-url, GET /health, GET /audio/<file>');
});
