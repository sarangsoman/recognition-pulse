// Drafts one short recognition message. The rules that decide WHO is due live in the page (plain code);
// this function only writes the words, from facts the page sends. Credentials are injected by Netlify AI Gateway.
import Anthropic from '@anthropic-ai/sdk';

const REASONS = new Set(['anniversary', 'silent', 'peers_not_me', 'birthday']);
const TONES = { warm: 'warm and personal', brief: 'brief and direct' };
const clip = (v, n) => String(v ?? '').replace(/[\r\n]+/g, ' ').slice(0, n);

export default async (req) => {
  if (req.method !== 'POST') return Response.json({ error: 'POST only' }, { status: 405 });
  if (!process.env.ANTHROPIC_BASE_URL) return Response.json({ error: 'ai-unavailable' }, { status: 503 });

  let body;
  try { body = await req.json(); } catch { return Response.json({ error: 'bad json' }, { status: 400 }); }
  const f = body?.facts ?? {};
  if (!REASONS.has(body?.reason) || !f.first) return Response.json({ error: 'bad request' }, { status: 400 });

  const facts = {
    first_name: clip(f.first, 40),
    role_level: clip(f.level, 20),
    reason: body.reason,
    years_at_company: Number.isFinite(f.years) ? Math.min(40, Math.max(0, f.years | 0)) : null,
    company_value_they_show: f.topValue ? clip(f.topValue, 30) : null,
    recognitions_from_peers_last_90_days: Number.isFinite(f.received) ? f.received | 0 : null,
    days_since_last_recognition: Number.isFinite(f.daysSince) ? f.daysSince | 0 : null,
    manager_note: f.note ? clip(f.note, 200) : null,
  };
  const tone = TONES[body.tone] ?? TONES.warm;

  try {
    const client = new Anthropic();
    const res = await client.messages.create({
      model: 'claude-sonnet-5-5',
      max_tokens: 200,
      system:
        'You write short recognition messages that a manager sends to a team member. ' +
        'Rules: 2 to 3 sentences, under 50 words, plain workplace English, no emojis, no hashtags, no greeting like "Dear". ' +
        'Use ONLY the facts provided. Never invent projects, results, numbers or events. ' +
        'If manager_note is present, build the message around it. ' +
        'Do not mention days since last recognition or that data was analysed. ' +
        'Return only the message text.',
      messages: [{ role: 'user', content: `Tone: ${tone}.\nFacts (JSON): ${JSON.stringify(facts)}` }],
    });
    const text = res.content.filter((b) => b.type === 'text').map((b) => b.text).join('').trim();
    if (!text) return Response.json({ error: 'empty' }, { status: 502 });
    return Response.json({ draft: text.slice(0, 600), source: 'ai' });
  } catch (e) {
    return Response.json({ error: 'ai-failed' }, { status: 502 });
  }
};

export const config = {
  path: '/api/draft',
  method: 'POST',
  rateLimit: { action: 'rate_limit', aggregateBy: ['ip', 'domain'], windowSize: 60, windowLimit: 12 },
};
