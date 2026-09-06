"use client";

import { useId, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

const LETTER = new RegExp("\\p{L}", "u");
const FIRST_PAGE_SIZE = 3;
const NEXT_PAGE_SIZE = 7;

function isShortEnglish(text: string, language?: string) {
  const english = language === "English" || language === "en"
    || (!language && /[a-z]/i.test(text) && !LETTER.test(text.replace(/[a-z]/gi, "")));
  return english && /[a-z]/i.test(text) && (text.match(/[a-z0-9]+(?:['’][a-z0-9]+)*/gi) ?? []).length <= 3;
}

type Presentation =
  | { status: "hidden" | "original" | "unavailable"; language: string }
  | { status: "translated"; language: string; text: string };

export type ReviewEvidenceRecord = {
  evidence_id?: string;
  text: string;
  source_type: string;
  source_field?: string;
  language?: string;
  translated_text?: string;
  relevance_reason?: string;
  visit_date?: string;
  matched_terms?: string[];
  is_verbatim?: boolean;
  presentation?: Presentation;
};

export default function ReviewEvidence({ reviews, language }: {
  reviews: ReviewEvidenceRecord[];
  language: string;
}) {
  const korean = language === "Korean";
  const [expanded, setExpanded] = useState(true);
  const [page, setPage] = useState(0);
  const contentId = useId();
  const visible = reviews.filter(review => {
    if (!review.is_verbatim || !LETTER.test(review.text.replace(/[\u1100-\u11ff\u3130-\u318f]/g, ""))) return false;
    const presentation = review.presentation;
    if (presentation?.status === "translated" && presentation.language === language && presentation.text.trim()) {
      return !isShortEnglish(presentation.text, presentation.language);
    }
    return !isShortEnglish(review.text, review.language);
  });

  const lastPage = Math.max(0, Math.ceil((visible.length - FIRST_PAGE_SIZE) / NEXT_PAGE_SIZE));
  const currentPage = Math.min(page, lastPage);
  const pageStart = currentPage === 0 ? 0 : FIRST_PAGE_SIZE + (currentPage - 1) * NEXT_PAGE_SIZE;
  const pageEnd = Math.min(visible.length, pageStart + (currentPage === 0 ? FIRST_PAGE_SIZE : NEXT_PAGE_SIZE));
  const pageReviews = visible.slice(pageStart, pageEnd);

  return (
    <section className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <h4 className="text-sm font-semibold text-slate-800">
        <button
          type="button"
          className="flex w-full items-center justify-between gap-2 py-1 text-left"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-controls={contentId}
          aria-label={expanded ? (korean ? "후기 숨기기" : "Hide reviews") : (korean ? "후기 보기" : "Show reviews")}
        >
          <span>{korean ? "환자 후기" : "Patient reviews"} ({visible.length})</span>
          {expanded ? <ChevronUp size={16} aria-hidden="true" /> : <ChevronDown size={16} aria-hidden="true" />}
        </button>
      </h4>
      <div id={contentId} hidden={!expanded} className="mt-2">
        {visible.length === 0 ? (
          <p className="text-sm text-slate-600">
            {korean ? "표시할 수 있는 후기가 없습니다." : "No comments available to display."}
          </p>
        ) : (
          <div className="space-y-3">
            {pageReviews.map((review, index) => {
              const presentation = review.presentation;
              const translated = presentation?.status === "translated"
                && presentation.language === language && presentation.text.trim().length > 0;
              const unavailable = presentation?.status === "unavailable";
              return (
                <article key={review.evidence_id ?? index} className="rounded-md bg-white p-3 text-sm text-slate-700">
                  <p className="mb-1 text-xs text-slate-500">
                    {translated
                      ? (korean ? "자동 번역 · 원문 확인 가능" : "Automatic translation · original available")
                      : unavailable
                        ? (korean ? "번역을 제공할 수 없어 원문을 표시합니다." : "Translation unavailable. Showing the original.")
                        : (korean ? "원문 후기" : "Original review")}
                    {review.visit_date ? ` · ${review.visit_date}` : ""}
                  </p>
                  <blockquote className="whitespace-pre-wrap break-words text-slate-900">
                    {translated ? presentation.text : review.text}
                  </blockquote>
                  {translated && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-blue-700">
                        {korean ? "원문 보기" : "Show original"}
                      </summary>
                      <blockquote className="mt-2 whitespace-pre-wrap break-words text-slate-900">
                        {review.text}
                      </blockquote>
                    </details>
                  )}
                </article>
              );
            })}
          </div>
        )}
        {visible.length > FIRST_PAGE_SIZE && (
          <nav className="mt-3 flex items-center justify-between text-xs" aria-label={korean ? "후기 페이지" : "Review pages"}>
            <button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)} className="rounded px-2 py-1 text-blue-700 disabled:text-slate-400">
              {korean ? "이전" : "Previous"}
            </button>
            <span>{pageStart + 1}–{pageEnd} / {visible.length}</span>
            <button type="button" disabled={currentPage === lastPage} onClick={() => setPage(currentPage + 1)} className="rounded px-2 py-1 text-blue-700 disabled:text-slate-400">
              {korean ? "다음" : "Next"}
            </button>
          </nav>
        )}
        <p className="mt-2 text-xs text-slate-600">
          {korean
            ? "후기는 환자의 경험이며 요청하신 서비스나 언어 지원을 보장하지 않습니다."
            : "Patient experiences do not establish service availability or staff language abilities."}
        </p>
      </div>
    </section>
  );
}
