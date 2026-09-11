import { describe, expect, it } from "vitest";
import { validateQuery } from "@/lib/server/query-validation";
import { MAX_PAPER_ID_CHARS, QUESTION_MAX_CHARS, QUESTION_MIN_CHARS } from "@/lib/shared/limits";

/** A backend paper id: a 64-character SHA-256 hex digest. */
const SHA256_ID = "0123456789abcdef".repeat(4);

describe("validateQuery", () => {
  it("trims the question and applies the default top_k of 5", () => {
    expect(validateQuery({ question: "  What is attention?  " })).toStrictEqual({
      question: "What is attention?",
      top_k: 5,
    });
  });

  it("accepts questions from 3 to 2000 characters after trimming", () => {
    expect(QUESTION_MIN_CHARS).toBe(3);
    expect(QUESTION_MAX_CHARS).toBe(2000);
    expect(validateQuery({ question: "abc" })?.question).toBe("abc");
    expect(validateQuery({ question: "q".repeat(2000) })?.question).toHaveLength(2000);
    expect(validateQuery({ question: "ab" })).toBeNull();
    expect(validateQuery({ question: "  ab  " })).toBeNull();
    expect(validateQuery({ question: "q".repeat(2001) })).toBeNull();
  });

  it("rejects a missing or non-string question", () => {
    expect(validateQuery({})).toBeNull();
    expect(validateQuery({ question: 42 })).toBeNull();
    expect(validateQuery({ question: ["abc"] })).toBeNull();
  });

  it("accepts an integer top_k from 1 to 5", () => {
    expect(validateQuery({ question: "abc", top_k: 1 })?.top_k).toBe(1);
    expect(validateQuery({ question: "abc", top_k: 3 })?.top_k).toBe(3);
    expect(validateQuery({ question: "abc", top_k: 5 })?.top_k).toBe(5);
  });

  it("rejects top_k outside 1..5, non-integers and numeric strings", () => {
    for (const bad of [0, 6, 2.5, "3"]) {
      expect(validateQuery({ question: "abc", top_k: bad }), `top_k ${JSON.stringify(bad)}`).toBeNull();
    }
  });

  it("accepts a paper_id in the backend's id alphabet", () => {
    expect(validateQuery({ question: "abc", paper_id: SHA256_ID })?.paper_id).toBe(SHA256_ID);
    expect(validateQuery({ question: "abc", paper_id: "arxiv_2106-09685" })?.paper_id).toBe("arxiv_2106-09685");
    const longest = "a".repeat(MAX_PAPER_ID_CHARS);
    expect(validateQuery({ question: "abc", paper_id: longest })?.paper_id).toBe(longest);
  });

  it("rejects a paper_id with spaces, slashes, over 128 characters, empty, or not a string", () => {
    const bad: unknown[] = ["abc def", "a/b", "a\\b", "a".repeat(MAX_PAPER_ID_CHARS + 1), "", 123, { id: SHA256_ID }];
    for (const paper_id of bad) {
      expect(validateQuery({ question: "abc", paper_id }), `paper_id ${JSON.stringify(paper_id)}`).toBeNull();
    }
  });

  it("treats a null or undefined paper_id as absent", () => {
    const expected = { question: "abc", top_k: 5 };
    expect(validateQuery({ question: "abc", paper_id: null })).toStrictEqual(expected);
    expect(validateQuery({ question: "abc", paper_id: undefined })).toStrictEqual(expected);
    expect(validateQuery({ question: "abc", paper_id: null })).not.toHaveProperty("paper_id");
  });

  it("drops unknown fields instead of forwarding them", () => {
    const result = validateQuery({
      question: "abc",
      top_k: 3,
      paper_id: SHA256_ID,
      model: "gpt-4o",
      stream: true,
      temperature: 2,
    });
    expect(result).toStrictEqual({ question: "abc", top_k: 3, paper_id: SHA256_ID });
    expect(result).not.toHaveProperty("model");
  });
});
