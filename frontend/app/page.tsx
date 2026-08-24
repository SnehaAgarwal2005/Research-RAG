"use client";

import { FormEvent, useState } from "react";

type Source = {
  paper?: string;
  page?: string | number;
  chunk?: string | number;
};

type ApiResponse = {
  answer?: string;
  sources?: Source[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data: ApiResponse = await response.json();

      setAnswer(data.answer || "No answer was returned.");
      setSources(data.sources || []);
    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to the RAG backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              📚 Research RAG Assistant
            </h1>

            <p className="mt-1 text-sm text-slate-400">
              Ask questions about research papers using hybrid retrieval.
            </p>
          </div>

          <div className="hidden rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-300 sm:block">
            FAISS + BM25 + RRF + Gemini
          </div>
        </div>
      </header>

      {/* Main */}
      <section className="mx-auto max-w-6xl px-6 py-12">
        {/* Hero */}
        <div className="mx-auto max-w-3xl text-center">
          <div className="mb-4 inline-flex rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-300">
            🔎 Hybrid Retrieval · 🤖 Grounded Generation
          </div>

          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Research papers,
            <span className="block text-slate-400">
              answered intelligently.
            </span>
          </h2>

          <p className="mx-auto mt-5 max-w-2xl text-lg leading-8 text-slate-400">
            Search your research documents using semantic and keyword
            retrieval, then generate answers grounded in the retrieved
            sources.
          </p>
        </div>

        {/* Question Box */}
        <form
          onSubmit={askQuestion}
          className="mx-auto mt-10 max-w-4xl"
        >
          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3 shadow-2xl">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="e.g. How does QLoRA reduce memory usage?"
              rows={3}
              className="w-full resize-none bg-transparent px-3 py-2 text-base text-white outline-none placeholder:text-slate-500"
            />

            <div className="flex items-center justify-between border-t border-slate-800 pt-3">
              <span className="px-3 text-xs text-slate-500">
                Answers are generated from your indexed research papers.
              </span>

              <button
                type="submit"
                disabled={loading}
                className="rounded-xl bg-white px-5 py-2.5 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Searching..." : "🔍 Ask"}
              </button>
            </div>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className="mx-auto mt-6 max-w-4xl rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
            ❌ {error}
          </div>
        )}

        {/* Answer */}
        {answer && (
          <section className="mx-auto mt-10 max-w-4xl">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-800">
                  🤖
                </div>

                <div>
                  <h3 className="font-semibold">Answer</h3>
                  <p className="text-xs text-slate-500">
                    Generated from retrieved research context
                  </p>
                </div>
              </div>

              <div className="whitespace-pre-wrap text-[15px] leading-7 text-slate-300">
                {answer}
              </div>
            </div>
          </section>
        )}

        {/* Sources */}
        {sources.length > 0 && (
          <section className="mx-auto mt-6 max-w-4xl">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
              <h3 className="mb-5 text-lg font-semibold">
                📚 Retrieved Sources
              </h3>

              <div className="space-y-3">
                {sources.map((source, index) => (
                  <div
                    key={`${source.paper}-${source.page}-${index}`}
                    className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                  >
                    <div className="flex items-start gap-3">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-800 text-xs font-semibold">
                        {index + 1}
                      </span>

                      <div>
                        <p className="font-medium text-slate-200">
                          {source.paper || "Unknown paper"}
                        </p>

                        <p className="mt-1 text-sm text-slate-500">
                          Page {source.page ?? "?"}
                          {" · "}
                          Chunk {source.chunk ?? "?"}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Pipeline */}
        <section className="mx-auto mt-16 max-w-5xl">
          <div className="mb-8 text-center">
            <h3 className="text-2xl font-bold">
              How the RAG Pipeline Works
            </h3>

            <p className="mt-2 text-slate-400">
              Multiple retrieval strategies are combined before generation.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {[
              ["01", "BGE Embeddings", "Convert document chunks into vectors."],
              ["02", "FAISS", "Semantic similarity search."],
              ["03", "BM25", "Keyword-based retrieval."],
              ["04", "RRF", "Combine retrieval rankings."],
              ["05", "Gemini", "Generate a grounded answer."],
            ].map(([number, title, description]) => (
              <div
                key={number}
                className="rounded-2xl border border-slate-800 bg-slate-900 p-5"
              >
                <div className="text-xs font-bold text-slate-500">
                  {number}
                </div>

                <h4 className="mt-3 font-semibold">
                  {title}
                </h4>

                <p className="mt-2 text-sm leading-6 text-slate-500">
                  {description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Technology Stack */}
        <section className="mx-auto mt-16 max-w-4xl">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <h3 className="text-lg font-semibold">
              ⚙️ Technology Stack
            </h3>

            <div className="mt-5 flex flex-wrap gap-2">
              {[
                "Python",
                "FastAPI",
                "Next.js",
                "TypeScript",
                "FAISS",
                "BM25",
                "BGE",
                "RRF",
                "Gemini",
                "Streamlit",
              ].map((technology) => (
                <span
                  key={technology}
                  className="rounded-full border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm text-slate-300"
                >
                  {technology}
                </span>
              ))}
            </div>
          </div>
        </section>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8 text-center text-sm text-slate-500">
        Research RAG Assistant · Hybrid Retrieval + Grounded Generation
      </footer>
    </main>
  );
}