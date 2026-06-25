# Translation Notes — SimCLR (arXiv:2002.05709v3)

## Translation Metadata

- **Paper:** A Simple Framework for Contrastive Learning of Visual Representations
- **Date processed:** 2026-06-24
- **Source format:** pdf-text (arXiv selectable-text PDF)
- **Total pages:** 20
- **Total source blocks:** 34 body text blocks mapped (+ appendices summarized)

## Terminology Decisions

| English | Chosen Chinese | Alternatives Considered | Rationale |
|---------|---------------|------------------------|-----------|
| contrastive learning | 对比学习 | 对比式学习 | Standard in Chinese ML literature |
| representation | 表征 | 表示 | 表征 more common in mainland Chinese ML papers |
| pretext task | 前置任务 | 代理任务、借口任务 | 前置任务 is the established term |
| projection head | 投影头 | 投影层 | 头 emphasizes it's a small add-on module |
| temperature parameter | 温度参数 | — | Direct translation, universally accepted |
| hard negative | 难负样本 | 困难负样本 | 难负样本 is more concise and common |
| memory bank | 记忆库 | 记忆银行 | 记忆库 is standard |
| fine-tuning | 微调 | 精调 | 微调 is standard |
| data augmentation | 数据增强 | 数据扩增 | Both acceptable; 增强 used in PyTorch docs |
| transfer learning | 迁移学习 | 转移学习 | 迁移学习 is standard |
| linear evaluation | 线性评估 | 线性探测 | 评估 more accurate than 探测 for protocol meaning |

## Extraction Confidence

- **Pages 1–11 (main body):** HIGH — text layer is clean, well-structured, selectable
- **Pages 12–20 (appendices):** HIGH — text layer clean, but tables and figures are not extractable as structured data from PDF text; pseudocode formatting may have minor artifacts
- **Equations:** FULLY EXTRACTED — Equation (1) NT-Xent loss preserved
- **Tables:** MANUALLY RECONSTRUCTED — numerical values verified against text extraction; table layouts reconstructed from text dump
- **Figures:** NOT EXTRACTED — images noted by page location in paper.md; actual figure crops require PDF rendering (not possible from text-only extraction)

## Block Coverage

| Section | Blocks | Status |
|---------|--------|--------|
| Abstract | S002 | FULL |
| §1 Introduction | S003–S007 | FULL |
| §2 Method | S008–S014 | FULL |
| §3 Data Augmentation | S015–S018 | FULL |
| §4 Architectures | S019–S021 | FULL |
| §5 Loss Functions | S022–S024 | FULL |
| §6 Comparison with SOTA | S025–S028 | FULL |
| §7 Related Work | S029–S032 | FULL |
| §8 Conclusion | S033 | FULL |
| Appendices A-C | S034 | SUMMARIZED (key tables extracted) |
| References | — | NOTED — full list in source PDF pp.9–11 |

## Known Issues / Caveats

1. **Figure images not cropped:** The text-layer PDF extraction cannot produce image crops. The `assets/` directory is created but empty. To get actual figure images, re-extract from the PDF using PyMuPDF/fitz or a PDF viewer.
2. **Multi-column reading order:** The pypdf text extraction follows the PDF's internal stream order, which for two-column layouts (used throughout the paper) may occasionally produce out-of-order fragments. Manual review was performed to rejoin cross-column text.
3. **Algorithm 1 pseudocode:** Preserved in English only — pseudocode is largely code and does not benefit from translation.
4. **References section:** Not translated — references are ~60 entries spanning pp.9–11. They are available in the original PDF and the extracted text JSON.
5. **Appendix B experimental details:** Summarized rather than fully translated — the full bilingual treatment of 8 supplementary pages would make the reader unwieldy. Key tables (B.1–B.5) and findings are included.

## Draft Mode

This is a **complete** reader for the main body (Sections 1–8 + Abstract). Appendices are summarized with key tables extracted. Figures are location-tagged but not cropped into assets/. If full appendix translation is needed, it can be added as a follow-up.
