# ============================================================================
# PROMETHEUS — Intelligence Layer: Multi-Provider AI (100% FREE)
# ============================================================================
"""
AI intelligence layer using a smart multi-provider routing strategy.
All providers are FREE tier — zero cost for trading analysis.

Architecture (priority order):
  1. Groq Cloud (FREE) → Llama 3.3 70B — best reasoning, blazing fast
  2. Google Gemini 2.0 Flash (FREE) → powerful backup, 1500 RPD
  3. Ollama Local → any model that fits your hardware — offline fallback
  4. FinBERT (local) → specialized financial sentiment scoring
  5. Sentence-Transformers (local) → historical pattern matching

Why this order:
  - Groq free tier gives 70B-class reasoning at zero cost (14,400 req/day)
  - Gemini 2.0 Flash is the best free cloud fallback
  - Ollama ensures system works fully offline (no internet needed)
  - FinBERT handles sentiment better than any general LLM
  - A typical trading day uses ~30-50 LLM calls — well under free limits

Setup:
  - Groq: Get free API key at https://console.groq.com/keys
  - Gemini: Get free API key at https://aistudio.google.com/apikey
  - Ollama: Install from https://ollama.ai/download, then ollama pull llama3.2:3b
"""

import json
import os
import time
from typing import Optional, Dict, List
from datetime import datetime

from prometheus.utils.logger import logger


# ============================================================================
# Shared Prompts
# ============================================================================

SYSTEM_PROMPT = (
    "You are a senior quantitative analyst specializing in Indian F&O markets "
    "(NSE/BSE). You provide data-driven market analysis with focus on NIFTY, "
    "BANKNIFTY, and stock options. Be concise, precise, and always express "
    "confidence levels. When uncertain, say so clearly. Always respond in valid JSON "
    "when asked for structured output."
)


def build_market_prompt(context: Dict) -> str:
    """Build the standard market analysis prompt used by all providers."""
    return f"""Analyze the following Indian stock market data and provide a directional bias.

MARKET DATA:
- Symbol: {context.get('symbol', 'NIFTY')}
- Spot Price: {context.get('spot_price', 'N/A')}
- India VIX: {context.get('vix', 'N/A')}
- Market Regime: {context.get('regime', 'N/A')}
- Trend Strength: {context.get('trend_strength', 'N/A')}
- OI Sentiment: {context.get('oi_sentiment', 'N/A')}
- PCR: {context.get('pcr', 'N/A')}
- Max Pain: {context.get('max_pain', 'N/A')}
- Recent Headlines: {context.get('headlines', 'None')}

Respond ONLY in this JSON format:
{{
    "direction": "bullish" or "bearish" or "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning": "1-2 sentence explanation",
    "risk_factors": ["list of key risks"],
    "suggested_action": "what to do"
}}"""


def build_trade_prompt(trade: Dict) -> str:
    """Build the standard trade explanation prompt."""
    return f"""Explain this trade decision in simple terms that a non-expert trader can understand:

Trade: {trade.get('action', 'N/A')} on {trade.get('symbol', 'N/A')}
Entry: Rs {trade.get('entry_price', 'N/A')}
Stop Loss: Rs {trade.get('stop_loss', 'N/A')}
Target: Rs {trade.get('target', 'N/A')}
Risk:Reward: {trade.get('risk_reward', 'N/A')}
Regime: {trade.get('regime', 'N/A')}
Signals: {trade.get('contributing_signals', [])}

Explain in 3-4 sentences:
1. What is the trade
2. Why the system is taking it
3. Where is the stop loss and why
4. What is the expected profit"""


def build_news_prompt(headlines: List[str]) -> str:
    """Build the standard news analysis prompt."""
    headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
    return f"""Analyze these news headlines for their impact on the Indian stock market (NIFTY/BANKNIFTY).

HEADLINES:
{headlines_text}

For each significant headline, assess:
1. Immediate market impact (bullish/bearish/neutral)
2. Affected sectors/instruments
3. Expected duration of impact (intraday/multi-day/structural)
4. Confidence level

Respond in this JSON format:
{{
    "overall_sentiment": "bullish" or "bearish" or "neutral",
    "overall_score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "key_events": [
        {{
            "headline": "summary",
            "impact": "bullish/bearish/neutral",
            "affected": ["NIFTY", "BANKNIFTY", etc],
            "duration": "intraday/multi-day/structural",
            "reasoning": "why"
        }}
    ],
    "action_bias": "what should traders do"
}}"""


def parse_json_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    for marker in ["```json", "```"]:
        if marker in response:
            start = response.index(marker) + len(marker)
            end = response.index("```", start) if "```" in response[start:] else len(response)
            try:
                return json.loads(response[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try finding JSON-like content
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = response.find(start_char)
        end = response.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

    logger.debug(f"Failed to parse LLM JSON response: {response[:200]}")
    return None


# ============================================================================
# Provider 1: Groq (FREE — Llama 3.3 70B)
# ============================================================================

class GroqProvider:
    """
    Groq Cloud — blazing fast inference of Llama 3.3 70B for FREE.

    Free tier limits (very generous for trading):
      - 30 requests/min
      - 14,400 requests/day
      - 6,000 tokens/min (input), 6,000 tokens/min (output)

    This gives you a 70B parameter model — real institutional-grade
    reasoning about market structure, news impact, and trade logic.

    Setup:
      1. Go to https://console.groq.com/keys
      2. Create a free API key
      3. Add to config/credentials.yaml under groq.api_key
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._last_request_time = 0.0
        self._min_interval = 2.0  # 30 RPM = 1 every 2s, stay safe

    def is_available(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_groq_api_key_here"

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response via Groq API (OpenAI-compatible)."""
        if not self.is_available():
            return None

        # Simple rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }

            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            self._last_request_time = time.time()

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]

            if resp.status_code == 429:
                logger.warning("Groq rate limited — falling back to next provider")
                return None

            logger.warning(f"Groq returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return None

    def generate_text(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate free-text response (no JSON mode)."""
        if not self.is_available():
            return None

        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            self._last_request_time = time.time()

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]

            if resp.status_code == 429:
                logger.warning("Groq rate limited — falling back to next provider")
                return None

            logger.warning(f"Groq returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        except Exception as e:
            logger.error(f"Groq text generation failed: {e}")
            return None


# ============================================================================
# Provider 2: Google Gemini 2.0 Flash (FREE)
# ============================================================================

class GeminiProvider:
    """
    Google Gemini 2.0 Flash — powerful free cloud LLM.

    Free tier:
      - 15 requests/min
      - 1,500 requests/day
      - 1M tokens/day

    Setup:
      1. Go to https://aistudio.google.com/apikey
      2. Create a free API key
      3. Add to config/credentials.yaml under gemini.api_key
    """

    def __init__(self, api_key: str = "", model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_gemini_api_key_here"

    def _init_client(self):
        if self._client:
            return True
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Gemini client initialized ({self.model})")
            return True
        except ImportError:
            logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
        return False

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate response from Gemini."""
        if not self.is_available():
            return None

        if not self._init_client():
            return None

        try:
            full_prompt = f"{system_prompt or SYSTEM_PROMPT}\n\n{prompt}"
            response = self._client.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return None


# ============================================================================
# Provider 3: Ollama Local (FREE — offline)
# ============================================================================

class OllamaProvider:
    """
    Local LLM via Ollama — runs any model that fits your hardware.

    Works completely offline. Best for:
      - No internet scenarios
      - Privacy (data never leaves your machine)
      - Unlimited requests

    Setup:
      1. Install Ollama: https://ollama.ai/download
      2. Pull a model: ollama pull llama3.2:3b (or qwen2.5:7b if you have GPU)
      3. It runs on localhost:11434 by default
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 90,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self._available = any(self.model.split(":")[0] in m for m in models)
                if not self._available:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
                return self._available
        except Exception:
            pass
        self._available = False
        return False

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generate a response from the local LLM."""
        if not self.is_available():
            logger.debug("Ollama not available, skipping")
            return None

        try:
            import requests
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt or SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code == 200:
                return resp.json().get("response", "")

            logger.warning(f"Ollama returned HTTP {resp.status_code}")
            return None

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None


# ============================================================================
# FinBERT Sentiment (local, finance-specialized)
# ============================================================================

class FinBERTSentiment:
    """
    Financial sentiment analysis using FinBERT.
    Runs locally via HuggingFace transformers — free, no API needed.

    FinBERT is specifically trained on financial text and outperforms
    general sentiment models on market-related content.

    Setup: pip install transformers torch
    First run will download the model (~440MB).
    """

    def __init__(self, cache_dir: str = "data/models/finbert"):
        self.model_name = "ProsusAI/finbert"
        self.cache_dir = cache_dir
        self._pipeline = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            os.makedirs(self.cache_dir, exist_ok=True)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                cache_dir=self.cache_dir,
                device=-1,  # CPU — change to 0 for GPU
            )
            logger.info("FinBERT model loaded successfully")
        except ImportError:
            logger.warning("transformers not installed. Run: pip install transformers torch")
        except Exception as e:
            logger.error(f"FinBERT load failed: {e}")

    def analyze(self, text: str) -> Dict:
        """Analyze a single text for financial sentiment."""
        self._load_model()
        if self._pipeline is None:
            return {"label": "neutral", "score": 0.5}

        try:
            result = self._pipeline(text[:512])[0]
            return {
                "label": result["label"],
                "score": round(result["score"], 4),
            }
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {"label": "neutral", "score": 0.5}

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in batch (more efficient)."""
        self._load_model()
        if self._pipeline is None:
            return [{"label": "neutral", "score": 0.5} for _ in texts]

        try:
            trimmed = [t[:512] for t in texts]
            results = self._pipeline(trimmed)
            return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]
        except Exception as e:
            logger.error(f"FinBERT batch analysis failed: {e}")
            return [{"label": "neutral", "score": 0.5} for _ in texts]

    def aggregate_sentiment(self, texts: List[str]) -> Dict:
        """Analyze multiple texts and return aggregate sentiment."""
        if not texts:
            return {"direction": "neutral", "score": 0.0, "confidence": 0.0}

        results = self.analyze_batch(texts)

        positive = sum(1 for r in results if r["label"] == "positive")
        negative = sum(1 for r in results if r["label"] == "negative")
        total = len(results)

        avg_score = sum(
            r["score"] * (1 if r["label"] == "positive" else -1 if r["label"] == "negative" else 0)
            for r in results
        ) / total

        if avg_score > 0.1:
            direction = "bullish"
        elif avg_score < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = abs(avg_score)

        return {
            "direction": direction,
            "score": round(avg_score, 3),
            "confidence": round(confidence, 3),
            "positive_count": positive,
            "negative_count": negative,
            "total": total,
        }


# ============================================================================
# Pattern Matcher (local embeddings)
# ============================================================================

class PatternMatcher:
    """
    Embedding-based historical pattern matching.
    Uses sentence-transformers to find similar market conditions in history.
    """

    def __init__(self, cache_dir: str = "data/models/embeddings"):
        self.model_name = "all-MiniLM-L6-v2"
        self.cache_dir = cache_dir
        self._model = None
        self._pattern_db: List[Dict] = []

    def _load_model(self):
        """Lazy load embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            os.makedirs(self.cache_dir, exist_ok=True)
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")

    def encode_market_state(self, state: Dict) -> Optional[list]:
        """Convert market state to embedding vector."""
        self._load_model()
        if self._model is None:
            return None

        text = self._state_to_text(state)

        try:
            embedding = self._model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def store_pattern(self, state: Dict, outcome: str, pnl: float):
        """Store a market pattern with its outcome for future matching."""
        embedding = self.encode_market_state(state)
        if embedding:
            self._pattern_db.append({
                "embedding": embedding,
                "state": state,
                "outcome": outcome,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat(),
            })

    def find_similar(self, current_state: Dict, top_k: int = 5) -> List[Dict]:
        """Find the most similar historical patterns to current market state."""
        if not self._pattern_db:
            return []

        current_embedding = self.encode_market_state(current_state)
        if current_embedding is None:
            return []

        import numpy as np

        current_vec = np.array(current_embedding)
        similarities = []

        for pattern in self._pattern_db:
            stored_vec = np.array(pattern["embedding"])
            similarity = np.dot(current_vec, stored_vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(stored_vec)
            )
            similarities.append((similarity, pattern))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, pattern in similarities[:top_k]:
            results.append({
                "similarity": round(float(sim), 4),
                "outcome": pattern["outcome"],
                "pnl": pattern["pnl"],
                "timestamp": pattern["timestamp"],
                "state": pattern["state"],
            })

        return results

    def _state_to_text(self, state: Dict) -> str:
        """Convert market state dict to descriptive text for embedding."""
        parts = []
        if "regime" in state:
            parts.append(f"Market regime is {state['regime']}.")
        if "trend_strength" in state:
            ts = state["trend_strength"]
            direction = "uptrend" if ts > 0.2 else "downtrend" if ts < -0.2 else "sideways"
            parts.append(f"Trend is {direction} with strength {abs(ts):.2f}.")
        if "vix" in state:
            vix = state["vix"]
            vol_level = "low" if vix < 14 else "medium" if vix < 20 else "high" if vix < 28 else "extreme"
            parts.append(f"Volatility is {vol_level} with VIX at {vix}.")
        if "pcr" in state:
            parts.append(f"Put-call ratio is {state['pcr']:.2f}.")
        if "oi_sentiment" in state:
            parts.append(f"OI sentiment is {state['oi_sentiment']}.")

        return " ".join(parts) if parts else "No market data available."


# ============================================================================
# Intelligence Engine — Smart Multi-Provider Router
# ============================================================================

class IntelligenceEngine:
    """
    Unified AI intelligence layer with smart provider routing.

    Priority chain (all FREE):
      1. Groq → Llama 3.3 70B (best reasoning, fastest cloud)
      2. Gemini → 2.0 Flash (powerful backup)
      3. Ollama → local model (offline fallback)
      4. FinBERT → financial sentiment (always local)
      5. Embeddings → pattern matching (always local)

    The engine automatically falls through the chain if a provider
    is unavailable or rate-limited. You always get a result.
    """

    def __init__(self, config: Dict):
        # Provider 1: Groq (primary — 70B reasoning for free)
        groq_cfg = config.get("groq", {})
        self.groq = GroqProvider(
            api_key=groq_cfg.get("api_key", ""),
            model=groq_cfg.get("model", "llama-3.3-70b-versatile"),
            temperature=groq_cfg.get("temperature", 0.3),
            max_tokens=groq_cfg.get("max_tokens", 2048),
        )

        # Provider 2: Gemini (secondary — powerful cloud backup)
        gemini_cfg = config.get("gemini", {})
        self.gemini = GeminiProvider(
            api_key=gemini_cfg.get("api_key", ""),
            model=gemini_cfg.get("model", "gemini-2.0-flash"),
        )

        # Provider 3: Ollama (tertiary — offline fallback)
        ollama_cfg = config.get("ollama", {})
        self.ollama = OllamaProvider(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_cfg.get("model", "llama3.2:3b"),
            temperature=ollama_cfg.get("temperature", 0.3),
            max_tokens=ollama_cfg.get("max_tokens", 2048),
        )

        # Local specialist models (always available)
        self.finbert = FinBERTSentiment(
            cache_dir=config.get("finbert", {}).get("cache_dir", "data/models/finbert")
        )
        self.patterns = PatternMatcher(
            cache_dir=config.get("embeddings", {}).get("cache_dir", "data/models/embeddings")
        )

        self._log_availability()

    def _log_availability(self):
        """Log which AI providers are available."""
        status = []
        if self.groq.is_available():
            status.append(f"Groq ({self.groq.model}) [PRIMARY]")
        if self.gemini.is_available():
            status.append(f"Gemini ({self.gemini.model})")
        if self.ollama.is_available():
            status.append(f"Ollama ({self.ollama.model})")
        status.append("FinBERT (on-demand)")
        status.append("Embeddings (on-demand)")

        logger.info(f"AI providers: {', '.join(status)}")

        if not self.groq.is_available() and not self.gemini.is_available():
            if not self.ollama.is_available():
                logger.warning(
                    "No LLM providers available! Set up at least one:\n"
                    "  - Groq (best): Get free key at https://console.groq.com/keys\n"
                    "  - Gemini: Get free key at https://aistudio.google.com/apikey\n"
                    "  - Ollama: Install from https://ollama.ai/download"
                )

    def _generate_with_fallback(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Try each provider in priority order until one succeeds."""
        # Try Groq first (70B, best quality)
        result = self.groq.generate(prompt, system_prompt)
        if result:
            logger.debug("LLM response from: Groq")
            return result

        # Try Gemini (powerful cloud backup)
        result = self.gemini.generate(prompt, system_prompt)
        if result:
            logger.debug("LLM response from: Gemini")
            return result

        # Try Ollama (local fallback)
        result = self.ollama.generate(prompt, system_prompt)
        if result:
            logger.debug("LLM response from: Ollama")
            return result

        logger.warning("All LLM providers failed — no analysis available")
        return None

    def _generate_text_with_fallback(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Try each provider for free-text output (no JSON mode)."""
        # Groq (free text mode)
        result = self.groq.generate_text(prompt, system_prompt)
        if result:
            return result

        # Gemini
        result = self.gemini.generate(prompt, system_prompt)
        if result:
            return result

        # Ollama
        result = self.ollama.generate(prompt, system_prompt)
        if result:
            return result

        return None

    def analyze_market_context(self, context: Dict) -> Optional[Dict]:
        """Run LLM market analysis — routes through provider chain."""
        prompt = build_market_prompt(context)
        response = self._generate_with_fallback(prompt)
        if response:
            return parse_json_response(response)
        return None

    def analyze_news_sentiment(self, headlines: List[str]) -> Dict:
        """Analyze news using FinBERT (fast, specialized) + LLM (deep reasoning)."""
        # FinBERT for quick sentiment scoring
        finbert_result = self.finbert.aggregate_sentiment(headlines)

        # LLM for deeper reasoning (via provider chain)
        llm_result = None
        if headlines:
            prompt = build_news_prompt(headlines)
            response = self._generate_with_fallback(prompt)
            if response:
                llm_result = parse_json_response(response)

        # Combine: FinBERT score + LLM reasoning
        combined = {
            "direction": finbert_result["direction"],
            "score": finbert_result["score"],
            "confidence": finbert_result["confidence"],
            "source": "finbert",
        }

        if llm_result:
            combined["llm_direction"] = llm_result.get("overall_sentiment", "neutral")
            combined["llm_reasoning"] = llm_result.get("action_bias", "")
            combined["key_events"] = llm_result.get("key_events", [])

            # If both agree, boost confidence
            if combined["direction"] == combined["llm_direction"]:
                combined["confidence"] = min(combined["confidence"] * 1.3, 1.0)
                combined["source"] = "finbert+llm"

        return combined

    def explain_trade(self, trade: Dict) -> str:
        """Generate plain-language trade explanation."""
        prompt = build_trade_prompt(trade)
        explanation = self._generate_text_with_fallback(prompt)
        if explanation:
            return explanation

        # Fallback: template-based explanation
        return self._template_explanation(trade)

    def find_historical_analogs(self, state: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar historical market conditions."""
        return self.patterns.find_similar(state, top_k)

    def record_trade_outcome(self, state: Dict, outcome: str, pnl: float):
        """Record a trade outcome for future pattern matching (learning)."""
        self.patterns.store_pattern(state, outcome, pnl)

    def _template_explanation(self, trade: Dict) -> str:
        """Fallback template when no LLM is available."""
        action = trade.get("action", "HOLD")
        symbol = trade.get("symbol", "NIFTY")
        entry = trade.get("entry_price", 0)
        sl = trade.get("stop_loss", 0)
        target = trade.get("target", 0)
        rr = trade.get("risk_reward", 0)

        if action == "HOLD":
            return f"No trade on {symbol}. Market conditions don't meet our confluence threshold."

        direction = "bullish (BUY CALL)" if "CE" in action else "bearish (BUY PUT)"
        return (
            f"Taking a {direction} position on {symbol}. "
            f"Entry near Rs {entry:.0f}, stop loss at Rs {sl:.0f}, target Rs {target:.0f}. "
            f"Risk:Reward ratio is 1:{rr:.1f}. "
            f"This trade is based on {len(trade.get('contributing_signals', []))} "
            f"converging signals from our analysis engine."
        )
