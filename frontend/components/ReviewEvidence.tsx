"use client";

import { forwardRef, useEffect, useId, useImperativeHandle, useRef, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

const FIRST_PAGE_SIZE = 3;
const NEXT_PAGE_SIZE = 7;
const ORIGINAL_PREVIEW_LENGTH = 480;

type Presentation =
  | { status: "hidden" | "original" | "unavailable"; language: string }
  | { status: "translated"; language: string; text: string };

export type ReviewEvidenceRecord = {
  evidence_id: string;
  place_id: string;
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
  review_source_sha256?: string;
  source_locator?: string;
  retrieval_roles?: string[];
};

export type ReviewCitation = {
  marker: number;
  place_id: string;
  evidence_id: string;
  original_excerpt: string;
  review_source_sha256?: string;
};

export type ReviewEvidenceHandle = {
  showReview: (evidenceId: string) => void;
};

function isDisplayableOriginal(review: ReviewEvidenceRecord, facilityId: string) {
  return review !== null && typeof review === "object"
    && review.place_id === facilityId
    && typeof review.evidence_id === "string" && review.evidence_id.length > 0
    && review.source_type === "verbatim_review" && review.is_verbatim === true
    && review.presentation?.status !== "hidden"
    && typeof review.text === "string" && review.text.trim().length > 0;
}

export function canLocateCitation(citation: ReviewCitation, facilityId: string, reviews: ReviewEvidenceRecord[]) {
  return citation !== null && typeof citation === "object"
    && Number.isSafeInteger(citation.marker) && citation.marker > 0
    && citation.place_id === facilityId
    && typeof citation.original_excerpt === "string" && citation.original_excerpt.length > 0
    && reviews.some(review => isDisplayableOriginal(review, facilityId)
      && review.evidence_id === citation.evidence_id
      && review.text.includes(citation.original_excerpt)
      && (!citation.review_source_sha256 || citation.review_source_sha256 === review.review_source_sha256));
}

const ReviewEvidence = forwardRef<ReviewEvidenceHandle, {
  reviews: ReviewEvidenceRecord[];
  language: string;
  facilityId: string;
}>(function ReviewEvidence({ reviews, language, facilityId }, ref) {
  const korean = language === "Korean";
  const [expanded, setExpanded] = useState(true);
  const [page, setPage] = useState(0);
  const [fullReviewIds, setFullReviewIds] = useState<Set<string>>(new Set());
  const [focusedReviewId, setFocusedReviewId] = useState<string | null>(null);
  const [focusRequest, setFocusRequest] = useState(0);
  const reviewElements = useRef(new Map<string, HTMLElement>());
  const contentId = useId();
  const visible = reviews.filter(review => isDisplayableOriginal(review, facilityId)).map(review => {
    const presentation = review.presentation;
    const translatedText = presentation?.status === "translated"
      && presentation.language === language
      && typeof presentation.text === "string" && presentation.text.trim().length > 0
      ? presentation.text : undefined;
    return { review, translatedText };
  });

  useImperativeHandle(ref, () => ({
    showReview(evidenceId) {
      const index = visible.findIndex(item => item.review.evidence_id === evidenceId);
      if (index < 0) return;
      setExpanded(true);
      setPage(index < FIRST_PAGE_SIZE ? 0 : 1 + Math.floor((index - FIRST_PAGE_SIZE) / NEXT_PAGE_SIZE));
      setFullReviewIds(previous => new Set(previous).add(evidenceId));
      setFocusedReviewId(evidenceId);
      setFocusRequest(previous => previous + 1);
    },
  }));

  useEffect(() => {
    if (!focusedReviewId) return;
    const element = reviewElements.current.get(focusedReviewId);
    element?.focus({ preventScroll: true });
    element?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [focusedReviewId, focusRequest]);

  const lastPage = Math.max(0, Math.ceil((visible.length - FIRST_PAGE_SIZE) / NEXT_PAGE_SIZE));
  const currentPage = Math.min(page, lastPage);
  const pageStart = currentPage === 0 ? 0 : FIRST_PAGE_SIZE + (currentPage - 1) * NEXT_PAGE_SIZE;
  const pageEnd = Math.min(visible.length, pageStart + (currentPage === 0 ? FIRST_PAGE_SIZE : NEXT_PAGE_SIZE));
  const pageReviews = visible.slice(pageStart, pageEnd);

  return (
    <section data-facility-id={facilityId} className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
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
            {pageReviews.map(({ review, translatedText }) => {
              const unavailable = review.presentation?.status === "unavailable"
                || (review.presentation?.status === "translated" && !translatedText);
              const fullOriginal = fullReviewIds.has(review.evidence_id);
              const originalPreview = Array.from(review.text).slice(0, ORIGINAL_PREVIEW_LENGTH).join("");
              const longOriginal = originalPreview.length < review.text.length;
              const originalId = `${contentId}-${review.evidence_id}`;
              const originalContent = (
                <>
                  <p className="mb-1 text-xs text-slate-500">
                    {korean ? "원문 후기" : "Original review"}
                    {review.visit_date ? ` · ${review.visit_date}` : ""}
                  </p>
                  <blockquote id={originalId} data-original-review className="whitespace-pre-wrap break-words text-slate-900">
                    {longOriginal && !fullOriginal ? originalPreview : review.text}
                  </blockquote>
                  {longOriginal && (
                    <button
                      type="button"
                      className="mt-2 py-1 text-blue-700"
                      aria-expanded={fullOriginal}
                      aria-controls={originalId}
                      onClick={() => setFullReviewIds(previous => {
                        const next = new Set(previous);
                        if (fullOriginal) next.delete(review.evidence_id);
                        else next.add(review.evidence_id);
                        return next;
                      })}
                    >
                      {fullOriginal ? (korean ? "미리보기" : "Show preview") : (korean ? "원문 전체 보기" : "Read full original")}
                    </button>
                  )}
                </>
              );
              return (
                <article
                  key={review.evidence_id}
                  data-evidence-id={review.evidence_id}
                  tabIndex={-1}
                  ref={element => {
                    if (element) reviewElements.current.set(review.evidence_id, element);
                    else reviewElements.current.delete(review.evidence_id);
                  }}
                  className="rounded-md bg-white p-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {translatedText ? (
                    <>
                      <p className="mb-1 text-xs text-slate-500">
                        {korean ? "자동 번역" : "Automatic translation"}
                        {review.visit_date ? ` · ${review.visit_date}` : ""}
                      </p>
                      <p data-review-translation className="whitespace-pre-wrap break-words text-slate-900">{translatedText}</p>
                      <details key={focusRequest} className="mt-2" open={focusedReviewId === review.evidence_id}>
                        <summary className="cursor-pointer text-blue-700">
                          {korean ? "원문 보기" : "Show original"}
                        </summary>
                        <div className="mt-2">{originalContent}</div>
                      </details>
                    </>
                  ) : originalContent}
                  {unavailable && (
                    <p className="mt-2 text-xs text-slate-500">{korean ? "번역을 제공할 수 없어 원문을 표시합니다." : "Translation unavailable. Showing the original."}</p>
                  )}
                </article>
              );
            })}
          </div>
        )}
        {visible.length < reviews.length && (
          <p className="mt-2 text-xs text-slate-600">{korean ? "출처를 확인할 수 없는 일부 후기는 표시하지 않습니다." : "Some reviews could not be shown because their source could not be verified."}</p>
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
});

export default ReviewEvidence;
