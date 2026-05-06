import { describe, expect, it } from "vitest";
import { loadSchema } from "../src/schema-loader.js";

function vectorIndexDimensions(schema: string): string[] {
  return Array.from(schema.matchAll(/HNSW DIMENSION (\d+) DIST COSINE/g))
    .map(match => match[1]);
}

describe("loadSchema", () => {
  it("renders 1024-dimensional vector indexes by default", () => {
    const dimensions = vectorIndexDimensions(loadSchema());
    expect(dimensions).toHaveLength(8);
    expect(dimensions.every(dim => dim === "1024")).toBe(true);
    expect(loadSchema()).not.toContain("__KONGBRAIN_EMBEDDING_DIMENSIONS__");
  });

  it("renders vector indexes using the configured embedding dimension", () => {
    const schema = loadSchema({ embeddingDimensions: 768 });
    const dimensions = vectorIndexDimensions(schema);
    expect(dimensions).toHaveLength(8);
    expect(dimensions.every(dim => dim === "768")).toBe(true);
  });

  it("falls back to 1024 for invalid embedding dimensions", () => {
    for (const invalid of [0, -1, 1.5, Number.NaN]) {
      const dimensions = vectorIndexDimensions(loadSchema({ embeddingDimensions: invalid }));
      expect(dimensions).toHaveLength(8);
      expect(dimensions.every(dim => dim === "1024")).toBe(true);
    }
  });
});
