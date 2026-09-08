const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");
const spaceApis = [
  "https://valerianfourel-seouldoctor-ncs-retriever.hf.space",
  "https://valerianfourel-seouldoctor.hf.space",
];
const attemptTimeoutMs = 90_000;

export function getApiBases(hostname: string): string[] {
  if (hostname === "seouldoc.io" || hostname === "www.seouldoc.io") {
    return spaceApis;
  }
  if (hostname.endsWith(".hf.space")) return [""];
  if (configuredApiUrl && configuredApiUrl !== "https://seouldoctor.onrender.com") {
    return [configuredApiUrl];
  }
  return hostname.endsWith(".vercel.app") ? spaceApis : [""];
}

const getErrorMessage = (body: unknown, status: number): string => {
  if (body && typeof body === "object" && "detail" in body) {
    if (typeof body.detail === "string") return body.detail;
  }
  return `Request failed with status ${status}.`;
};

function hasConversationBody(body: unknown, path: string): boolean {
  if (!body || typeof body !== "object" || !("state" in body)) return false;
  if (!body.state || typeof body.state !== "object" || Array.isArray(body.state)) return false;
  return path !== "/chat" || (
    "response" in body && typeof body.response === "string" &&
    (!("results" in body) || Array.isArray(body.results))
  );
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const conversationRequest = normalizedPath === "/chat" || normalizedPath === "/set_travel_preference";
  const bases = getApiBases(typeof window === "undefined" ? "" : window.location.hostname);
  const attempts = conversationRequest ? bases : bases.slice(0, 1);
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (typeof window !== "undefined") {
    try {
      const savedConsent = localStorage.getItem("cookieConsent");
      if (savedConsent) headers.set("X-Cookie-Consent", savedConsent);
    } catch {
      // Some privacy modes disable storage. The API then uses denied defaults.
    }
  }

  for (let index = 0; index < attempts.length; index += 1) {
    const base = attempts[index];
    init.signal?.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort(init.signal?.reason);
    init.signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(), attemptTimeoutMs);
    let retryable = true;
    try {
      const response = await fetch(`${base}${normalizedPath}`, {
        ...init,
        credentials: "same-origin",
        headers,
        signal: controller.signal,
      });
      retryable = response.status >= 500;
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) throw new Error(getErrorMessage(body, response.status));
      retryable = true;
      if (body === null || (conversationRequest && !hasConversationBody(body, normalizedPath))) {
        throw new Error("The server returned an incomplete response. Please try again.");
      }
      return body as T;
    } catch (error) {
      init.signal?.throwIfAborted();
      if (!retryable) throw error;
      if (index === attempts.length - 1) {
        throw new Error("SeoulDoc is temporarily unavailable. Please try again in a moment.");
      }
    } finally {
      clearTimeout(timer);
      init.signal?.removeEventListener("abort", abort);
    }
  }
  throw new Error("No SeoulDoc API is configured.");
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, { method: "POST", body: JSON.stringify(body) });
}
