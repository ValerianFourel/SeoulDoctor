const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

const getApiUrl = (path: string): string => {
  if (!configuredApiUrl) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured.");
  }

  const baseUrl = configuredApiUrl.replace(/\/$/, "");
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}${normalizedPath}`;
};

const getErrorMessage = (body: unknown, status: number): string => {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  }

  return `Request failed with status ${status}.`;
};

export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (typeof window !== "undefined") {
    try {
      const savedConsent = localStorage.getItem("cookieConsent");
      if (savedConsent) headers.set("X-Cookie-Consent", savedConsent);
    } catch {
      // Some privacy modes disable storage. The API then uses denied defaults.
    }
  }

  const response = await fetch(getApiUrl(path), {
    ...init,
    credentials: "include",
    headers,
  });

  const body = (await response.json().catch(() => null)) as unknown;

  if (!response.ok) {
    throw new Error(getErrorMessage(body, response.status));
  }

  return body as T;
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
