import type { SignatureSet } from "./signatures";

/**
 * Modules that ship with Slidecraft, so a map is readable the moment it loads.
 *
 * A single predicted gene inherits all of that gene's error, and per-gene
 * accuracy from H&E is modest. Averaging a module's genes — each standardised
 * first, so an abundant one like COL1A1 cannot drown the rest — leaves the
 * shared signal and averages the independent noise down. The difference is
 * stark in practice: single genes look like static, and the same genes read as
 * a module show mucosa, submucosa and muscle wall.
 *
 * These carry no weights. A weight says how specific a gene is to a type, and
 * that is a number you get from an atlas, not from judgement — which is what
 * `scripts/signatures_from_cellxgene.py` is for. What these are is a curated
 * grouping: markers whose spatial arrangement is already known, so a map that
 * is wrong is recognisable as wrong rather than merely colourful.
 *
 * Kept in step with IBD_PANEL in scripts/predict_expression.py. A module whose
 * genes the map does not carry is hidden rather than shown empty, so the same
 * list serves an eight-gene map and a whole-transcriptome one.
 */
export const BUILTIN_SIGNATURES: SignatureSet = {
  source: "Slidecraft built-in modules",
  organism: "Homo sapiens",
  signatures: [
    {
      name: "Epithelium",
      genes: ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT20", "VIL1"].map(gene => ({ gene, weight: 1 })),
    },
    {
      // Mature absorptive colonocyte. These fall away in active disease, so
      // their absence reads as informatively as their presence.
      name: "Colonocyte (mature)",
      genes: ["CA1", "CA2", "AQP8", "SLC26A3", "GUCA2A", "GUCA2B", "MS4A12", "CEACAM7",
              "HMGCS2", "SELENBP1"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Goblet / mucus",
      genes: ["MUC2", "TFF3", "FCGBP", "CLCA1", "ZG16", "SPINK4", "AGR2", "ITLN1",
              "REG4"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Crypt / proliferation",
      genes: ["LGR5", "OLFM4", "ASCL2", "MKI67", "SOX9", "REG1A", "REG1B",
              "REG3A"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Antimicrobial / Paneth",
      genes: ["LYZ", "DEFA5", "DEFA6", "PLA2G2A", "PI3", "SLPI", "LCN2",
              "DMBT1"].map(gene => ({ gene, weight: 1 })),
    },
    {
      // S100A8/A9 are faecal calprotectin itself.
      name: "Neutrophil / calprotectin",
      genes: ["S100A8", "S100A9", "S100A12", "FCGR3B", "CXCL8",
              "CSF3R"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "T cells",
      genes: ["PTPRC", "CD3D", "CD3E", "CD2", "CD8A", "CD4", "FOXP3", "IL7R", "CCL5",
              "GZMA", "GZMB"].map(gene => ({ gene, weight: 1 })),
    },
    {
      // Plasma cells dominate inflamed IBD mucosa.
      name: "B / plasma cells",
      genes: ["MS4A1", "CD79A", "JCHAIN", "MZB1", "DERL3", "XBP1"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Myeloid",
      genes: ["CD68", "CD14", "CD163", "ITGAX", "C1QA", "TYROBP", "AIF1", "HLA-DRA",
              "CD74"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Cytokines / drug targets",
      genes: ["TNF", "IL1B", "IL6", "IL17A", "IL23A", "IL12B", "IFNG", "OSM", "OSMR",
              "JAK1", "JAK2", "TYK2", "S1PR1", "ITGA4", "ITGB7", "IL10", "IL11",
              "TNFAIP3"].map(gene => ({ gene, weight: 1 })),
    },
    {
      // CXCL13/CCL19/CCL21 mark the tertiary lymphoid structures of chronicity.
      name: "Chemokines / lymphoid",
      genes: ["CXCL9", "CXCL10", "CXCL11", "CXCL13", "CCL19", "CCL21", "CXCL5",
              "CXCL1"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Stroma / fibrosis",
      genes: ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL6A1", "FN1", "ACTA2", "TAGLN",
              "DES", "PDGFRA", "PDGFRB", "THY1", "VIM", "MMP1", "MMP3", "MMP9", "TIMP1",
              "TGFB1", "POSTN", "FAP"].map(gene => ({ gene, weight: 1 })),
    },
    {
      // MADCAM1 is the vedolizumab target; the rest is vessel identity.
      name: "Vascular",
      genes: ["PECAM1", "VWF", "CDH5", "MADCAM1", "ICAM1", "VCAM1", "ACKR1",
              "PLVAP"].map(gene => ({ gene, weight: 1 })),
    },
    {
      name: "Neuroendocrine",
      genes: ["CHGA", "CHGB", "PYY", "S100B", "UCHL1"].map(gene => ({ gene, weight: 1 })),
    },
  ],
};
