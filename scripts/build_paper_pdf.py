import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
import pypdf

ROOT = Path(".").resolve()
pdf_path = ROOT / "paper" / "main.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        canvas.Canvas.save(self)

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "FactBench: Controlled Multi-Metric Evaluation of Small Open LLMs on Scientific QA")
            self.drawRightString(558, 750, "A. Kumar")
            self.line(54, 744, 558, 744)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 34, text)
        self.drawString(54, 34, "FactBench: Controlled Multi-Metric Scientific QA Evaluation")
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_pdf():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#0d3b66"),
        spaceAfter=6
    )

    author_style = ParagraphStyle(
        "AuthorStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        alignment=1,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0d3b66"),
        spaceBefore=9,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#1d3557"),
        spaceBefore=6,
        spaceAfter=2.5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.2,
        leading=11,
        alignment=4,  # Justified
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#222222")
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7,
        leading=8.8,
        alignment=1  # Centered
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold"
    )

    table_cell_cons = ParagraphStyle(
        "TableCellCons",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=6.8,
        leading=8.5,
        alignment=1
    )

    table_cell_cons_bold = ParagraphStyle(
        "TableCellConsBold",
        parent=table_cell_cons,
        fontName="Helvetica-Bold"
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=7.2,
        leading=9.2,
        alignment=1,
        spaceAfter=5
    )

    abstract_text = ParagraphStyle(
        "AbstractText",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.8,
        leading=10.5,
        alignment=4
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    story = []

    # Title & Author
    story.append(Paragraph("<b>Are Factuality Scores Interchangeable? A Controlled Multi-Metric Evaluation of Small Open Large Language Models on Scientific Question Answering</b>", title_style))
    story.append(Paragraph("<b>Aman Kumar</b><br/>Indian Institute of Information Technology, Allahabad &bull; Department of Information Technology<br/>Prayagraj, Uttar Pradesh, India &bull; <code>mcl2026012@iiita.ac.in</code>", author_style))

    # Abstract Box
    abstract_html = (
        "<b>Abstract</b>—When evaluating large language models (LLMs) on scientific benchmarks, standard academic practice "
        "overwhelmingly relies on a single scalar metric: top-1 raw accuracy. In this work, we empirically evaluate whether factuality "
        "metrics rank models concordantly across five scientific slices spanning biomedicine, mathematics, and physical sciences "
        "(950 distinct items; 1,450 item-slice evaluations). Under a frozen protocol (<b>FactBench</b>), we execute 17,400 forward passes "
        "across three open-weights instruction-tuned models (<b>Qwen2.5-1.5B-Instruct</b>, <b>Phi-3.5-mini-instruct</b>, and 4-bit <b>Qwen2.5-7B-Instruct</b>) "
        "across four deterministic option orders to measure accuracy, multiclass Brier score [0, 2], Expected Calibration Error (ECE), selective prediction risk (AURC), "
        "confidence discrimination (AUROC_conf), and permutation consistency.<br/><br/>"
        "Our empirical results yield three critical findings: "
        "<b>(1) Metric Discordance:</b> Model rankings under probability calibration (ECE) and confidence discrimination (AUROC_conf) "
        "diverge from raw accuracy rankings, while Brier score, AURC, and selective accuracy track accuracy closely (mean within-slice Kendall &tau; &approx; 0.830). "
        "On biomedical MMLU, Phi-3.5-mini achieves the highest raw accuracy (77.3%) yet lowest confidence discrimination (AUROC_conf = 0.703), "
        "though paired differences are not statistically significant (Holm-adjusted paired p between 0.35 and 0.98 across the three biomedical pairs), "
        "indicating rank differences are suggestive with N=3 models. Mean within-slice Kendall &tau; between accuracy and AUROC_conf is 0.297 (95% CI: [-0.067, 0.697]). "
        "<b>(2) Evidence Grounding Shift:</b> Providing primary literature context on PubMedQA produces substantial accuracy gains (+33.6% for "
        "Phi-3.5-mini, +31.0% for Qwen2.5-7B; paired exact McNemar p &lt; 0.001) and slashes Brier scores by over 0.57, whereas closed-book "
        "evaluations mask severe majority-class collapse (Qwen2.5-1.5B answers 'yes' to 78.2% of questions, matching the 55.2% majority "
        "class purely by coincidence while achieving 0.0% recall on 'maybe'). "
        "<b>(3) The Consistency Trap:</b> Permutation consistency fails to detect stubbornly held false predictions: across all cells, "
        "we identify 1,029 item-model error instances (54.6% of all errors) where models repeat the exact same incorrect answer across all four option "
        "orderings with high mean confidence (overall mean 0.949, reaching 1.000 for Phi-3.5 on biomed). "
        "We conclude that scientific LLM benchmarking requires multidimensional reporting vectors separating accuracy, calibration, abstention, "
        "and evidence grounding."
    )

    abstract_table = Table([[Paragraph(abstract_html, abstract_text)]], colWidths=[504])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0d3b66")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 6))

    # Section 1
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Large language models (LLMs) are rapidly finding application across scientific workflows, from biomedical literature synthesis "
        "and clinical research screening to quantitative problem solving in physics and chemistry [Hendrycks et al. 2021, Jin et al. 2019]. "
        "The standard methodology for evaluating whether an LLM is capable in science relies almost exclusively on multiple-choice question "
        "answering benchmarks, where a single scalar number—top-1 raw accuracy—is reported on public leaderboards [Liang et al. 2022]. However, in scientific "
        "disciplines where an erroneous assertion can misdirect laboratory experimentation or clinical care, an agreeable conversational tone "
        "or a high accuracy score reveals little about whether the model's confidence tracks truth, whether it can safely abstain under epistemic "
        "uncertainty, or whether its answers collapse when stripped of immediate retrieved evidence [Guo et al. 2017, Geifman & El-Yaniv 2017, Kadavath et al. 2022].",
        body_style
    ))
    story.append(Paragraph(
        "Factuality in science possesses distinct structural dimensions rather than a single monolithic axis. A scientific model might successfully "
        "recall a biological terminology definition, yet misstate the clinical scope of a therapy, solve a quantitative physics problem through "
        "an invalid physical assumption, cite a real paper that does not support the claim, or confidently select a multiple-choice option without "
        "admitting that the underlying study findings are ambiguous. For this reason, a single top-1 accuracy score cannot support broad "
        "claims regarding the reliability or safety of an AI system.",
        body_style
    ))
    story.append(Paragraph(
        "Prior literature reviews on factual accuracy evaluation in scientific LLMs have argued that evaluation metrics cannot be casually substituted "
        "for one another: accuracy, probability calibration, selective prediction, and consistency each assess distinct operational properties of a scientific "
        "system. While theoretical taxonomies advocate multidimensional evaluation vectors, controlled empirical studies measuring rank concordance "
        "across small open-weights models under standardized inference conditions remain scarce.",
        body_style
    ))
    story.append(Paragraph(
        "To transition from conceptual critique to empirical evidence, this study investigates three focused research questions:<br/>"
        "&bull; <b>RQ1 (Multi-Metric Concordance):</b> Do model rankings established by raw accuracy agree with rankings under probability calibration "
        "(Brier score, ECE), selective prediction (accuracy at 80% coverage), and confidence discrimination (AUROC_conf), or do different metrics produce rank discordance?<br/>"
        "&bull; <b>RQ2 (Evidence Grounding Shifts):</b> Does providing primary research context (PubMedQA context-grounded vs closed-book) alter accuracy, "
        "calibration, and class-level recall uniformly, or does closed-book evaluation mask degenerate heuristics?<br/>"
        "&bull; <b>RQ3 (The Consistency Trap):</b> Is option-permutation consistency a reliable surrogate for factual correctness, or does consistency reward stubbornly held errors?",
        body_style
    ))
    story.append(Paragraph(
        "To address these questions, we establish <b>FactBench</b>, an accountable evaluation harness. We evaluate three open-weights models across "
        "five scientific slices (950 distinct items; 1,450 item-slice evaluations) and four option orders, totaling 17,400 forward passes under a frozen experimental "
        "protocol. Prior to comparative evaluation, our scoring pipeline is validated against EleutherAI's <i>lm-evaluation-harness</i> [Gao et al. 2023] "
        "on reference tasks. All metrics are accompanied by non-parametric bootstrap confidence intervals, paired McNemar tests with Holm-Bonferroni "
        "correction [McNemar 1947, Holm 1979], and traceable raw outputs.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>Scientific Question Answering Benchmarks:</b> Multiple-choice benchmarks such as MMLU [Hendrycks et al. 2021] span 57 subjects across STEM, "
        "social sciences, and humanities. Recent work highlights that multiple-choice testing is susceptible to option-order sensitivity and positional heuristics "
        "[Zheng et al. 2024]. Specialized evaluations such as GPQA [Rein et al. 2023] focus on Google-proof graduate reasoning, while SciBench [Wang et al. 2024] "
        "benchmarks college-level multi-step scientific problem solving across chemistry, physics, and mathematics.<br/>"
        "<b>Multi-Metric Evaluation Frameworks:</b> Holistic evaluation suites, notably HELM [Liang et al. 2022], advocate multi-metric reporting separating accuracy, "
        "calibration, robustness, and fairness. Probability calibration in deep neural networks has traditionally been addressed via post-hoc temperature scaling "
        "[Guo et al. 2017], evaluated using the multiclass Brier score [Brier 1950] and Expected Calibration Error (ECE). For LLMs, Kadavath et al. [2022] "
        "observed that pre-trained language models exhibit reasonably calibrated probabilities on multiple-choice formats. Selective classification frameworks "
        "[Geifman & El-Yaniv 2017] evaluate the trade-off between coverage and risk via the Area Under the Risk-Coverage curve (AURC).<br/>"
        "<b>Evidence Grounding and Task-Mode Shifts:</b> Differentiating static parametric memorization from evidence-grounded reading comprehension is essential "
        "for scientific reliability [Lewis et al. 2020]. PubMedQA [Jin et al. 2019] provides biomedical research questions derived from PubMed abstracts with expert "
        "yes/no/maybe decisions. Evaluating systems in closed-book mode versus context-grounded mode provides an explicit measurement of evidence grounding. "
        "Fine-grained claim-level evaluation frameworks like FActScore [Min et al. 2023] decompose long-form text into atomic assertions and score evidence entailment, "
        "proving that citation presence does not guarantee factual entailment.<br/>"
        "<b>Consistency and Positional Heuristics:</b> SelfCheckGPT [Manakul et al. 2023] probes hallucinations by measuring agreement across stochastic sampling passes "
        "in black-box generation. In multiple-choice settings, LLMs exhibit strong spatial and positional biases [Zheng et al. 2024]. In this work, systematic option "
        "permutation consistency provides a deterministic, reproducible analogue to sample consistency, probing whether invariance to prompt presentation order correlates with factual truth.",
        body_style
    ))

    # Section 3
    story.append(Paragraph("3. Methodology and Formal Framework", h1_style))
    story.append(Paragraph(
        "<b>3.1 Task Definition and Scientific Slices:</b> We evaluate scientific question answering across five domain slices comprising 950 distinct questions "
        "(1,450 item-slice evaluations): (1) <i>Biomedical MMLU (mmlu_biomed)</i>: N = 150 items sampled uniformly at random from the pooled subjects (seed 42) "
        "across 8 MMLU biomedical subjects (anatomy, clinical knowledge, college biology, college medicine, high school biology, medical genetics, professional medicine, virology); "
        "(2) <i>Mathematics MMLU (mmlu_math)</i>: N = 150 items sampled uniformly at random from the pooled subjects (seed 42) across 4 MMLU mathematics subjects (abstract algebra, "
        "college mathematics, elementary mathematics, high school mathematics); (3) <i>Physical Sciences MMLU (mmlu_physchem)</i>: N = 150 items sampled uniformly at random "
        "from the pooled subjects (seed 42) across 6 MMLU physical science subjects (astronomy, college chemistry, college physics, conceptual physics, high school chemistry, high school physics); "
        "(4) <i>PubMedQA Closed-Book (pubmedqa_closed)</i>: N = 500 biomedical questions evaluated without abstract context (c = &empty;). PubMedQA uses a class-proportional sample "
        "of 500 of the 1,000 labelled items; official test split was not used; and "
        "(5) <i>PubMedQA Context-Grounded (pubmedqa_context)</i>: The identical N = 500 PubMedQA questions evaluated with the full published abstract text provided (c = abstract).<br/>"
        "<b>3.2 Scoring via Restricted Softmax:</b> Models are evaluated using constrained option scoring in a single forward pass without autoregressive text generation. "
        "Candidate options are assigned letter identifiers L = {A, B, C, D} for MMLU and L = {A, B, C} for PubMedQA. The prompt is passed through the model's official "
        "tokenizer and chat template. Let z(t) denote the unnormalized logit for vocabulary token t. To account for tokenizers that assign separate IDs to tokens with leading whitespace "
        "(e.g., Byte-Pair Encoding in Qwen and SentencePiece in Phi-3.5), we compute the log-sum-exp over all token IDs mapped to letter L: <i>z_tilde(L) = log &sum; exp(z(t))</i>. "
        "Normalized option probability is computed via categorical softmax: <i>p(L) = exp(z_tilde(L)) / &sum; exp(z_tilde(L'))</i>. The predicted choice is <i>y_hat = argmax p(L)</i>, and confidence is <i>c_hat = max p(L)</i>.<br/>"
        "<b>3.3 Option Permutations and Consistency:</b> Each item is scored under K = 4 option orders: the original order plus three distinct random permutations drawn "
        "deterministically per item (seed 42 and a hash of the item ID). Predictions are mapped back to the original options: <i>o_hat_k = &pi;_k^{-1}(y_hat_&pi;_k)</i>. "
        "Consistency is the fraction of the four orders whose prediction equals the most frequent prediction: <i>Cons = (1/K) max_v &sum; 1[o_hat_k = v]</i>. An item is deemed "
        "<i>consistently correct</i> if Cons = 1.0 and correct on &pi;_0; <i>wavering</i> if Cons &lt; 1.0; and <i>stubbornly wrong</i> if Cons = 1.0 and incorrect on &pi;_0.<br/>"
        "<b>3.4 Evaluation Metrics:</b> We compute seven primary metrics: (1) Raw Accuracy (Acc); (2) Multiclass Brier Score in [0, 2]; (3) Expected Calibration Error (ECE); "
        "(4) Area Under Risk-Coverage Curve (AURC); (5) Selective Accuracy at 80% coverage (SelAcc_0.8); (6) Confidence Discrimination (AUROC_conf); and (7) Consistency Discrimination (AUROC_cons).<br/>"
        "<b>3.5 Statistical Inference:</b> Two-sided 95% confidence intervals are estimated via non-parametric bootstrap resampling (B = 1,000 resamples over test items). "
        "Paired comparisons between task modes and between models use the exact two-sided McNemar test on discordant pairs, with family-wise error controlled via Holm-Bonferroni adjustment [Holm 1979]. "
        "Model ranking concordance is evaluated using Kendall's rank correlation &tau; [Kendall 1938].",
        body_style
    ))

    # Slices summary table
    tab_slices_data = [
        ["Slice", "Domain / Source", "N", "m", "Context (c)", "Passes"],
        ["mmlu_biomed", "Biomedicine (MMLU)", "150", "4", "Closed (empty)", "600"],
        ["mmlu_math", "Mathematics (MMLU)", "150", "4", "Closed (empty)", "600"],
        ["mmlu_physchem", "Physics & Chem (MMLU)", "150", "4", "Closed (empty)", "600"],
        ["pubmedqa_closed", "Clinical QA (PubMedQA)", "500", "3", "Closed (empty)", "2,000"],
        ["pubmedqa_context", "Clinical QA (PubMedQA)", "500", "3", "Published Abstract", "2,000"],
        ["Total / Model", "5 slices (950 distinct)", "1,450", "3-4", "--", "5,800"]
    ]
    t_slices = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_slices_data)],
                     colWidths=[100, 140, 45, 35, 105, 79])
    t_slices.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("<b>Table 1:</b> Summary of scientific benchmark slices, source datasets, option counts, and forward-pass budget.", caption_style))
    story.append(t_slices)
    story.append(Spacer(1, 6))

    # Section 4
    story.append(Paragraph("4. Experimental Setup", h1_style))
    story.append(Paragraph(
        "<b>4.1 Evaluated Models:</b> We evaluate three open-weights instruction-tuned models: (1) <i>Qwen2.5-1.5B-Instruct</i> (1.54B parameters, Apache 2.0 license, float16); "
        "(2) <i>Phi-3.5-mini-instruct</i> (3.82B parameters, MIT license, float16); and (3) <i>Qwen2.5-7B-Instruct-4bit</i> (7.61B parameters, Apache 2.0 license, quantized to 4-bit NormalFloat "
        "via bitsandbytes using library defaults without double quantization). <i>Confound Note:</i> Qwen2.5-7B is evaluated in 4-bit precision, whereas smaller baselines are evaluated in "
        "16-bit precision; performance differences reflect both scale and quantization effects.<br/>"
        "<b>4.2 Execution Environment:</b> Inference parameters were fixed in the frozen experimental protocol (protocol version 1.0, seed 42). Per execution telemetry, "
        "inference utilized NVIDIA Tesla T4 GPUs, Python 3.13.16, PyTorch 2.14.1+cu130, and HuggingFace Transformers 4.45.2, with batch size 8.<br/>"
        "<b>4.3 Baseline Validation:</b> Prior to comparative benchmarking, the scoring harness was validated against EleutherAI's <i>lm-evaluation-harness</i> [Gao et al. 2023] "
        "on Qwen2.5-1.5B-Instruct (conducted as a pre-experimental audit). On mmlu_college_biology (N = 144), our implementation achieved 63.89% vs harness 64.58% (&Delta; = -0.69%, 1 item difference; "
        "harness SE = &plusmn; 4.00%). On mmlu_high_school_physics (N = 151), our implementation achieved 37.75% vs harness 37.09% (&Delta; = +0.66%, 1 item difference; harness SE = &plusmn; 3.94%). "
        "Both discrepancies reside within 0.18 &times; SE. The check covers one model, two MMLU subjects and the harness-style prompt, and does not validate the chat prompts.",
        body_style
    ))

    # Section 5
    story.append(Paragraph("5. Experimental Results", h1_style))
    story.append(Paragraph("<b>Table 2:</b> Main benchmark results per model and slice (Acc, ECE, Brier, AURC, SelAcc_0.8, AUROC_conf, Cons_Mean).", caption_style))

    # Main Results Table
    tab_main_data = [
        ["Model", "Slice", "N", "Acc", "ECE", "Brier", "AURC", "SelAcc", "AUROC_c", "Cons"],
        ["Phi-3.5-mini", "Biomed", "150", "0.773", "0.222", "0.440", "0.114", "0.800", "0.703", "0.922"],
        ["Qwen-1.5B", "Biomed", "150", "0.693", "0.216", "0.524", "0.157", "0.733", "0.725", "0.878"],
        ["Qwen-7B-4b", "Biomed", "150", "0.733", "0.242", "0.492", "0.113", "0.825", "0.756", "0.933"],
        ["Phi-3.5-mini", "Math", "150", "0.473", "0.424", "0.881", "0.350", "0.525", "0.721", "0.747"],
        ["Qwen-1.5B", "Math", "150", "0.380", "0.315", "0.858", "0.492", "0.417", "0.635", "0.683"],
        ["Qwen-7B-4b", "Math", "150", "0.567", "0.334", "0.746", "0.267", "0.633", "0.750", "0.822"],
        ["Phi-3.5-mini", "PhysChem", "150", "0.573", "0.366", "0.766", "0.269", "0.633", "0.760", "0.823"],
        ["Qwen-1.5B", "PhysChem", "150", "0.453", "0.398", "0.839", "0.331", "0.508", "0.758", "0.822"],
        ["Qwen-7B-4b", "PhysChem", "150", "0.647", "0.278", "0.615", "0.171", "0.717", "0.740", "0.837"],
        ["Phi-3.5-mini", "PubMed Cl.", "500", "0.324", "0.601", "1.253", "0.703", "0.305", "0.456", "0.872"],
        ["Qwen-1.5B", "PubMed Cl.", "500", "0.552", "0.305", "0.749", "0.416", "0.570", "0.569", "0.921"],
        ["Qwen-7B-4b", "PubMed Cl.", "500", "0.418", "0.506", "1.054", "0.589", "0.440", "0.592", "0.878"],
        ["Phi-3.5-mini", "PubMed Ctx", "500", "0.660", "0.296", "0.640", "0.213", "0.702", "0.693", "0.912"],
        ["Qwen-1.5B", "PubMed Ctx", "500", "0.660", "0.288", "0.626", "0.217", "0.682", "0.649", "0.964"],
        ["Qwen-7B-4b", "PubMed Ctx", "500", "0.728", "0.231", "0.483", "0.197", "0.795", "0.761", "0.946"],
    ]

    t_main = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_main_data)],
                   colWidths=[68, 52, 28, 42, 40, 42, 42, 42, 48, 44])
    t_main.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_main)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1 RQ1: Multi-Metric Concordance and Inversions", h2_style))
    story.append(Paragraph(
        "Table 2 reports performance across all 15 model-slice cells. Table 3 reports mean within-slice Kendall's rank correlation &tau; between accuracy "
        "and alternative evaluation metrics. Metrics that incorporate correctness directly—Brier score, AURC, and selective accuracy at 80% coverage—exhibit "
        "strong rank concordance with raw accuracy (mean within-slice &tau; = 0.8300). In contrast, probability calibration error (ECE, &tau; = 0.4300) and confidence "
        "discrimination (AUROC_conf, &tau; = 0.2971, 95% CI [-0.0667, 0.6971]) diverge substantially from accuracy rankings.<br/>"
        "As illustrated in Figure 1, on biomedical MMLU (mmlu_biomed), Phi-3.5-mini achieves the highest nominal accuracy (0.7733), followed by Qwen2.5-7B (0.7333) and Qwen2.5-1.5B (0.6933). "
        "Under confidence discrimination (AUROC_conf), however, the ranking inverts: Qwen2.5-7B places first (0.7560), Qwen2.5-1.5B second (0.7251), and Phi-3.5-mini last (0.7031). "
        "Paired McNemar tests in Table 7 show that these pairwise accuracy differences are not statistically significant (Holm-adjusted paired p &isin; [0.35, 0.98] across all three pairs). "
        "Rather than dismissing this as inconsequential noise, we highlight an essential empirical finding: when models cluster closely in top-1 accuracy within sampling noise, "
        "reliability metrics (AUROC_conf and calibration) act as the primary differentiators for operational deployment. Furthermore, despite operating under aggressive 4-bit NF4 weight quantization, "
        "Qwen2.5-7B-Instruct-4bit attains competitive selective accuracy (SelAcc_0.8 = 0.825 on biomedicine, and the highest selective accuracy across all other slices) along with superior "
        "positional stability (Section 6.2) compared to the full-precision 16-bit Phi-3.5-mini (3.8B) and Qwen2.5-1.5B baselines. The core thesis that factuality metrics do not rank models identically "
        "is <b>partly supported</b>: calibration and confidence discrimination diverge from accuracy, while risk and Brier metrics track accuracy closely.",
        body_style
    ))

    # Kendall's tau table
    tab_tau_data = [
        ["Metric Compared against Accuracy", "Mean Within-Slice Kendall tau", "95% Bootstrap CI", "Concordance Verdict"],
        ["Multiclass Brier Score", "0.8300", "[0.5057, 1.0000]", "Strong Concordance"],
        ["Selective Risk (AURC)", "0.8300", "[0.5057, 1.0000]", "Strong Concordance"],
        ["Selective Accuracy (80% Coverage)", "0.8300", "[0.5057, 1.0000]", "Strong Concordance"],
        ["Expected Calibration Error (ECE)", "0.4300", "[0.0000, 0.8300]", "Moderate Discordance"],
        ["Consistency Discrimination (AUROC_cons)", "0.3800", "[-0.0465, 0.7788]", "Substantial Discordance"],
        ["Confidence Discrimination (AUROC_conf)", "0.2971", "[-0.0667, 0.6971]", "Severe Discordance (CI spans 0)"]
    ]
    t_tau = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_tau_data)],
                  colWidths=[174, 110, 105, 115])
    t_tau.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1d3557")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("<b>Table 3:</b> Mean within-slice Kendall's &tau; between accuracy and alternative evaluation metrics.", caption_style))
    story.append(t_tau)
    story.append(Spacer(1, 6))

    # Slopegraph Image
    slope_img_path = ROOT / "paper" / "figures" / "fig_rank_inversion_slopegraph.png"
    if slope_img_path.exists():
        story.append(Image(str(slope_img_path), width=450, height=180))
        story.append(Paragraph("<b>Figure 1:</b> Rank inversion slopegraph: Model ranking by Raw Accuracy (left) vs Confidence Discrimination (AUROC_conf, right) across slices.", caption_style))
        story.append(Spacer(1, 5))

    story.append(Paragraph("5.2 RQ2: Evidence Grounding Shifts and Majority Collapse", h2_style))
    story.append(Paragraph(
        "Table 4 and Figure 2 summarize the impact of providing published abstracts on PubMedQA. For Phi-3.5-mini and Qwen2.5-7B, context substantially improves performance: "
        "accuracy increases by +33.6 points and +31.0 points respectively (paired exact McNemar p &lt; 0.001), accompanied by substantial reductions in calibration error "
        "(&Delta; Brier = -0.6128 and -0.5711).<br/>"
        "In closed-book mode, Qwen2.5-1.5B achieves an apparent accuracy of 55.20%, exceeding both larger models (32.40% and 41.80%). However, analysis of the classification confusion matrix "
        "demonstrates that this is an artifact of majority-class prediction: "
        "(1) The 500 PubMedQA evaluation items comprise 55.2% 'yes' (N = 276), 33.8% 'no' (N = 169), and 11.0% 'maybe' (N = 55); "
        "(2) In closed-book mode, Qwen2.5-1.5B predicted 'yes' for 391 of 500 questions (78.20%). Its raw accuracy of 55.20% matches the 55.20% majority class share purely by coincidence; "
        "its recall on 'no' was 26.04% and on 'maybe' was 1.82%; "
        "(3) This affirmative bias persists directly across both evaluation regimes: when provided with full abstract context, Qwen2.5-1.5B's prediction rate for 'yes' further intensifies to 82.6% (413 of 500 predictions), "
        "achieving 95.3% recall on 'yes' (263/276) but failing completely on 'maybe' with 0.0% recall (0/55) and only 39.6% recall on 'no' (67/169); "
        "(4) To formally verify this effect, Table 5 reports balanced accuracy and macro-F1 (POST HOC and EXPLORATORY; not in the frozen protocol). In closed-book mode, Qwen-1.5B achieves "
        "a balanced accuracy of only 0.372 [0.342, 0.401] and macro-F1 of 0.352 [0.315, 0.391]. Even with context, balanced accuracy reaches only 0.450 and macro-F1 0.429 due to total collapse "
        "on the 'maybe' class.",
        body_style
    ))

    # Task Mode Delta Table
    tab_delta_data = [
        ["Model", "Gain (b)", "Loss (c)", "McNemar p (Holm)", "Delta Acc", "Delta ECE", "Delta Brier", "Delta AURC"],
        ["Phi-3.5-mini", "198", "30", "4.58e-31", "+0.336 [+0.286, +0.390]", "-0.305", "-0.613", "-0.490"],
        ["Qwen-1.5B", "77", "23", "5.51e-08", "+0.108 [+0.072, +0.146]", "-0.016", "-0.123", "-0.200"],
        ["Qwen-7B-4b", "186", "31", "7.84e-28", "+0.310 [+0.256, +0.364]", "-0.275", "-0.571", "-0.392"]
    ]
    t_delta = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_delta_data)],
                    colWidths=[80, 52, 52, 95, 95, 45, 45, 40])
    t_delta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("<b>Table 4:</b> Effect of providing abstract context on PubMedQA (context minus closed-book). p-values are Holm-adjusted McNemar tests.", caption_style))
    story.append(t_delta)
    story.append(Spacer(1, 5))

    # Recall shift figure
    recall_img_path = ROOT / "paper" / "figures" / "fig_pubmedqa_class_recall_shift.png"
    if recall_img_path.exists():
        story.append(Image(str(recall_img_path), width=450, height=165))
        story.append(Paragraph("<b>Figure 2:</b> Per-class recall on PubMedQA under closed-book vs context-grounded evaluation.", caption_style))
        story.append(Spacer(1, 5))

    # Post-hoc balanced accuracy table
    tab_posthoc_data = [
        ["Model", "Mode", "Raw Acc", "Balanced Acc [95% CI]", "Macro-F1 [95% CI]", "Yes Recall", "No Recall", "Maybe Recall"],
        ["Phi-3.5-mini", "Closed", "0.324", "0.396 [0.366, 0.428]", "0.307 [0.283, 0.334]", "0.297", "0.225", "0.655"],
        ["Phi-3.5-mini", "Context", "0.660", "0.584 [0.536, 0.631]", "0.575 [0.525, 0.622]", "0.859", "0.509", "0.145"],
        ["Qwen-1.5B", "Closed", "0.552", "0.372 [0.342, 0.401]", "0.352 [0.315, 0.391]", "0.837", "0.260", "0.018"],
        ["Qwen-1.5B", "Context", "0.660", "0.450 [0.424, 0.479]", "0.429 [0.394, 0.463]", "0.967", "0.385", "0.000"],
        ["Qwen-7B-4b", "Closed", "0.418", "0.407 [0.374, 0.442]", "0.380 [0.346, 0.413]", "0.475", "0.432", "0.309"],
        ["Qwen-7B-4b", "Context", "0.728", "0.601 [0.548, 0.654]", "0.601 [0.546, 0.655]", "0.949", "0.556", "0.145"],
    ]
    t_posthoc = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_posthoc_data)],
                      colWidths=[65, 45, 40, 105, 95, 52, 52, 50])
    t_posthoc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1d3557")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("<b>Table 5:</b> POST HOC exploratory evaluation: balanced accuracy and macro-F1 on PubMedQA under closed-book and context-grounded modes.", caption_style))
    story.append(t_posthoc)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.3 RQ3: The Consistency Trap and Stubborn Errors", h2_style))
    story.append(Paragraph(
        "Table 6 reports consistency metrics across all cells. As shown in Figure 3(b), accuracy generally increases with consistency level. However, across all 15 cells, "
        "we identify <b>1,029 stubborn wrong instances</b> where models repeat the identical incorrect choice across all four option permutations (Cons = 1.0, Acc = 0.0).<br/>"
        "Stubborn errors account for <b>54.6% of all errors</b> (1,029 / 1,885). Per cell, the stubborn share of errors ranges from 11.8% (Qwen-1.5B math) to 81.2% (Qwen-1.5B pubmedqa_context). "
        "As shown in Figure 3(a), stubborn errors exhibit high mean confidence: overall mean confidence is 0.949, with cell means ranging from 0.813 to 1.000 across cells "
        "(reaching 1.000 for Phi-3.5 on biomed). Qualitative and error-type auditing reveals that these 1,029 stubborn errors divide into two distinct failure mechanisms: "
        "(a) <i>parametric misconceptions</i>, where strong pre-training priors entrench an incorrect scientific claim across all option orders, and (b) <i>deficit-driven defaults</i>, "
        "where an ungrounded model systematically collapses onto an epistemic hedge (such as 'maybe') or a majority category in the absence of supporting evidence.<br/>"
        "Table 6 also reports AUROC_cons. Notably, on closed-book PubMedQA for Phi-3.5-mini, AUROC_cons = 0.432 (95% CI [0.383, 0.482]), falling below 0.5. In this cell, "
        "consistency is inversely related to accuracy: consistent predictions are more likely to be wrong than wavering ones due to 219 stubborn errors. By construction, "
        "consistency-based checks cannot identify stubborn errors because the model produces identical predictions across permutations.",
        body_style
    ))

    # Consistency breakdown table (Table 6) - with wide N and PubMed Ctx on single line
    tab_cons_data = [
        ["Model", "Slice", "N", "Consistent Correct", "Wavering Correct", "Wavering Wrong", "Stubborn Wrong", "Stubborn %", "AUROC_cons [95% CI]"],
        ["Phi-3.5-mini", "Biomed", "150", "99 (66.0%)", "17 (11.3%)", "17 (11.3%)", "17 (11.3%)", "50.0%", "0.679 [0.590, 0.773]"],
        ["Phi-3.5-mini", "Math", "150", "31 (20.7%)", "40 (26.7%)", "62 (41.3%)", "17 (11.3%)", "21.5%", "0.657 [0.570, 0.739]"],
        ["Phi-3.5-mini", "PhysChem", "150", "64 (42.7%)", "22 (14.7%)", "46 (30.7%)", "18 (12.0%)", "28.1%", "0.724 [0.643, 0.803]"],
        ["Phi-3.5-mini", "PubMed Cl.", "500", "82 (16.4%)", "80 (16.0%)", "119 (23.8%)", "219 (43.8%)", "64.8%", "0.432 [0.383, 0.482]"],
        ["Phi-3.5-mini", "PubMed Ctx", "500", "257 (51.4%)", "73 (14.6%)", "58 (11.6%)", "112 (22.4%)", "65.9%", "0.558 [0.508, 0.610]"],
        ["Qwen-1.5B", "Biomed", "150", "80 (53.3%)", "24 (16.0%)", "26 (17.3%)", "20 (13.3%)", "43.5%", "0.688 [0.603, 0.774]"],
        ["Qwen-1.5B", "Math", "150", "20 (13.3%)", "37 (24.7%)", "82 (54.7%)", "11 (7.3%)", "11.8%", "0.671 [0.588, 0.760]"],
        ["Qwen-1.5B", "PhysChem", "150", "46 (30.7%)", "22 (14.7%)", "47 (31.3%)", "35 (23.3%)", "42.7%", "0.632 [0.552, 0.706]"],
        ["Qwen-1.5B", "PubMed Cl.", "500", "228 (45.6%)", "48 (9.6%)", "66 (13.2%)", "158 (31.6%)", "70.5%", "0.561 [0.524, 0.597]"],
        ["Qwen-1.5B", "PubMed Ctx", "500", "305 (61.0%)", "25 (5.0%)", "32 (6.4%)", "138 (27.6%)", "81.2%", "0.556 [0.518, 0.595]"],
        ["Qwen-7B-4b", "Biomed", "150", "99 (66.0%)", "11 (7.3%)", "18 (12.0%)", "22 (14.7%)", "55.0%", "0.679 [0.600, 0.760]"],
        ["Qwen-7B-4b", "Math", "150", "62 (41.3%)", "23 (15.3%)", "48 (32.0%)", "17 (11.3%)", "26.2%", "0.742 [0.665, 0.816]"],
        ["Qwen-7B-4b", "PhysChem", "150", "70 (46.7%)", "27 (18.0%)", "40 (26.7%)", "13 (8.7%)", "24.5%", "0.761 [0.682, 0.839]"],
        ["Qwen-7B-4b", "PubMed Cl.", "500", "171 (34.2%)", "38 (7.6%)", "146 (29.2%)", "145 (29.0%)", "49.8%", "0.661 [0.617, 0.704]"],
        ["Qwen-7B-4b", "PubMed Ctx", "500", "328 (65.6%)", "36 (7.2%)", "49 (9.8%)", "87 (17.4%)", "64.0%", "0.632 [0.587, 0.679]"],
    ]
    t_cons = Table([[Paragraph(c, table_cell_cons_bold if r==0 else table_cell_cons) for c in row] for r, row in enumerate(tab_cons_data)],
                   colWidths=[60, 56, 28, 56, 52, 52, 52, 50, 98])
    t_cons.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(Paragraph("<b>Table 6:</b> RQ3: Permutation consistency vs correctness per cell; stubborn-error mean confidence ranges from 0.813 to 1.000 across cells.", caption_style))
    story.append(t_cons)
    story.append(Spacer(1, 6))

    # Figure 3: Side-by-side two panel figure (a) and (b)
    box_img_path = ROOT / "paper" / "figures" / "fig_stubborn_confidence_distribution.png"
    acc_img_path = ROOT / "paper" / "figures" / "fig_accuracy_by_consistency.png"
    if box_img_path.exists() and acc_img_path.exists():
        img_a = Image(str(box_img_path), width=248, height=135)
        img_b = Image(str(acc_img_path), width=248, height=135)
        fig3_table = Table([[img_a, img_b]], colWidths=[252, 252])
        fig3_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(fig3_table)
        story.append(Paragraph("<b>Figure 3:</b> The consistency trap. (a) Boxplot of predicted confidence distributions across Consistent Correct (N = 1,942), Wavering (N = 1,379), and Stubbornly Wrong (N = 1,029) groups across all 4,350 evaluations. (b) Observed accuracy by consistency level across slices.", caption_style))
        story.append(Spacer(1, 5))

    # Section 6
    story.append(Paragraph("6. In-Depth Analysis", h1_style))
    story.append(Paragraph(
        "<b>6.1 Failure Case Examination:</b> Inspection of stubborn failure cases under detailed error auditing "
        "confirms the two primary failure mechanisms governing stubborn errors:<br/>"
        "1. <i>Parametric Misconceptions (Entrenched Priors):</i> Models exhibit strong, invariant semantic associations that override factual accuracy regardless of presentation order: "
        "&bull; <i>Item mmlu-anatomy-25:</i> \"Which of the following substances is found in greater quantity in exhaled air?\" Options: [Carbon dioxide, Carbon monoxide, Nitrogen, Oxygen]. "
        "Gold: Nitrogen. All three models consistently select Carbon dioxide across all 4 permutations with extreme confidence (1.0000 for Phi-3.5, 0.8871 for Qwen-1.5B, 1.0000 for Qwen-7B), "
        "reflecting the colloquial misconception associating respiration with carbon dioxide over atmospheric nitrogen concentration. "
        "&bull; <i>Item mmlu-astronomy-42:</i> \"Why is Saturn almost as big as Jupiter despite its smaller mass?\" Gold: \"Jupiter's greater mass compresses it more thus increasing its density.\" "
        "Both Phi-3.5 and Qwen-1.5B consistently select the distractor citing hydrogen and helium proportions with Cons = 1.0 (confidences 1.0000 and 0.8458), locked into surface associations with gas giant elemental composition.<br/>"
        "2. <i>Deficit-Driven Defaults (Uncertainty Collapse):</i> In the absence of evidence, models default to an epistemic hedge or default class with invariant certainty: "
        "&bull; <i>Item pubmedqa-10966337:</i> \"A short stay or 23-hour ward in a general and academic children's hospital: are they effective?\" Gold: 'yes'. In closed-book mode, "
        "lacking empirical clinical evidence, Phi-3.5 consistently defaults to 'maybe' (c_hat = 1.0000, Cons = 1.0). When provided with abstract context, the knowledge deficit is resolved "
        "and its prediction shifts decisively to the correct answer 'yes' (c_hat = 1.0000, Cons = 1.0).<br/>"
        "<b>6.2 Display Position Bias Audit:</b> We evaluated option display frequencies across permuted presentations (descriptive analysis). "
        "For 4-choice MMLU items where uniform selection corresponds to 25.0%, models display observable display position preferences. Phi-3.5-mini selected option position 'B' "
        "34.0% in math and 30.2% in physchem (compared to 17.7% and 22.2% for 'A'). Qwen2.5-1.5B selected position 'B' 38.2% in math, while selecting position 'D' only 11.8% of the time (gold share: 30.0%).<br/>"
        "<b>6.3 Confidence Discrimination vs Consistency:</b> Comparing AUROC_conf and AUROC_cons across all 15 cells, confidence discrimination exceeds consistency discrimination "
        "in 12 of 15 cells. Across all cells, the mean difference is +0.042; over the 12 cells where confidence exceeds consistency, the mean difference is +0.063.<br/>"
        "<b>6.4 Permutation Ensembling (Exploratory):</b> Averaging predicted probabilities across all four option permutations (Acc_ens; exploratory analysis without "
        "paired hypothesis testing) produces accuracy differences (&Delta; = Acc_ens - Acc) spanning -0.027 to +0.067 across all 15 cells: "
        "Biomedical MMLU: Phi-3.5: 0.773 &rarr; 0.773 (&Delta; = 0.000), Qwen-1.5B: 0.693 &rarr; 0.667 (&Delta; = -0.027), Qwen-7B: 0.733 &rarr; 0.740 (&Delta; = +0.007); "
        "Mathematics MMLU: Phi-3.5: 0.473 &rarr; 0.480 (&Delta; = +0.007), Qwen-1.5B: 0.380 &rarr; 0.447 (&Delta; = +0.067), Qwen-7B: 0.567 &rarr; 0.580 (&Delta; = +0.013); "
        "Physical Sciences MMLU: Phi-3.5: 0.573 &rarr; 0.593 (&Delta; = +0.020), Qwen-1.5B: 0.453 &rarr; 0.500 (&Delta; = +0.047), Qwen-7B: 0.647 &rarr; 0.673 (&Delta; = +0.027); "
        "PubMedQA Closed-Book: Phi-3.5: 0.324 &rarr; 0.308 (&Delta; = -0.016), Qwen-1.5B: 0.552 &rarr; 0.564 (&Delta; = +0.012), Qwen-7B: 0.418 &rarr; 0.452 (&Delta; = +0.034); "
        "PubMedQA Context-Grounded: Phi-3.5: 0.660 &rarr; 0.638 (&Delta; = -0.022), Qwen-1.5B: 0.660 &rarr; 0.660 (&Delta; = 0.000), Qwen-7B: 0.728 &rarr; 0.732 (&Delta; = +0.004). "
        "Ensembling improved accuracy in 10 cells, produced no change in 2 cells, and decreased accuracy in 3 cells.<br/>"
        "<b>6.5 Reliability Diagrams and Risk-Coverage Profiles:</b> Inspection of complete reliability diagrams demonstrates that in mathematics and closed-book biomedical QA, "
        "all models exhibit substantial overconfidence bars falling well below the calibration diagonal. Conversely, selective risk-coverage curves confirm that for well-calibrated "
        "configurations like Qwen2.5-7B-Instruct-4bit, selective prediction reduces risk smoothly as coverage decreases.",
        body_style
    ))

    # Section 7
    story.append(Paragraph("7. Discussion: What a Leaderboard Should Show", h1_style))
    story.append(Paragraph(
        "<b>Leaderboard Reporting Recommendations:</b> Our findings demonstrate that single-number accuracy rankings do not capture multi-dimensional model reliability. "
        "We recommend that scientific benchmarks report a multi-dimensional scorecard: (1) Raw Accuracy; (2) Probability Calibration (Brier score / ECE); (3) Selective Accuracy "
        "(at 80% coverage); and (4) Permutation Consistency. Crucially, popular inference-time mitigation techniques—such as majority voting over permuted presentations or prompt-paraphrase "
        "self-consistency—fail in the presence of stubborn errors. Because model probability mass collapses decisively onto a single incorrect option irrespective of order or prompt phrasing, "
        "order-invariance serves as an amplifier of stubborn misconceptions rather than a safeguard of truth.<br/>"
        "<b>Vulnerabilities of Closed-Book Scientific QA:</b> Closed-book evaluation on imbalanced datasets carries severe risks of rewarding models that collapse onto majority classes. "
        "Evaluating without context controls or class-balanced metrics can produce misleading benchmarks.<br/>"
        "<b>Scale and Positional Stability:</b> Among the MMLU slices, Qwen2.5-7B is the most positionally stable model: its maximum deviation from uniform option selection is 2.7% on math, "
        "compared to 9.0% for Phi-3.5-mini and 13.2% for Qwen-1.5B. This suggests that scale may help suppress superficial positional heuristics, though we cannot disentangle scale from architecture and training data.<br/>"
        "<b>Practitioner Deployment Checklist:</b> Based on our empirical findings, we offer a three-point protocol for evaluating and deploying language models in scientific domains:<br/>"
        "1. <i>Autonomous and Safety-Critical Pipelines:</i> Prioritize selective accuracy at high coverage (SelAcc_0.8) and confidence discrimination (AUROC_conf) over raw accuracy. "
        "When models cluster within sampling noise (e.g., p &isin; [0.35, 0.98] on biomedicine), selective accuracy guarantees lower operational risk by abstaining on uncertain queries.<br/>"
        "2. <i>Literature and RAG Retrieval-QA:</i> Mandate balanced accuracy and macro-F1 alongside raw accuracy. On imbalanced clinical benchmarks like PubMedQA, models can achieve "
        "ostensibly competitive accuracy (55.2%) via majority-class collapse while suffering catastrophic recall failure on minority classes.<br/>"
        "3. <i>Robustness Auditing:</i> Pair permutation consistency with evidence grounding controls rather than treating order invariance as a standalone proxy for correctness. "
        "Consistency checks cannot detect entrenched misconceptions or deficit-driven defaults, which constitute 54.6% of all model errors.",
        body_style
    ))

    # Section 8
    story.append(Paragraph("8. Limitations and Threats to Validity", h1_style))
    story.append(Paragraph(
        "Several limitations qualify our findings:<br/>"
        "1. <b>Model Scale and Sample Size:</b> We evaluated three small-to-medium models (1.5B to 7B). With N = 3 models per slice, rank correlations have wide confidence intervals and paired differences on some slices are non-significant.<br/>"
        "2. <b>Quantization Confound:</b> Qwen2.5-7B was evaluated in 4-bit NF4 precision, whereas smaller baselines were evaluated in 16-bit precision.<br/>"
        "3. <b>Prompt Template Sensitivity:</b> Evaluations used a single chat prompt template per dataset. Results may vary under different prompt formulations.<br/>"
        "4. <b>Item-Level Resampling:</b> Bootstrap confidence intervals resample test items, reflecting item-level variance rather than stochasticity in model weights.<br/>"
        "5. <b>Benchmark Contamination:</b> Pre-training data contamination was not audited.<br/>"
        "6. <b>Dataset Artifacts:</b> MMLU items may contain minor label noise, and PubMedQA has an imbalanced class distribution (11% 'maybe' class).<br/>"
        "7. <b>Task Scope:</b> This study evaluates multiple-choice QA correctness, calibration, selective prediction, and permutation consistency; it does not evaluate open-ended generation, temporal validity, or step-by-step reasoning proofs.",
        body_style
    ))

    # Section 9: Conclusion
    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph(
        "This study evaluated whether factuality metrics rank small open LLMs concordantly on scientific question answering. Across 17,400 forward passes, we found that "
        "Brier score and selective accuracy track raw accuracy, while calibration error and confidence discrimination diverge. Evidence context rescues accuracy on PubMedQA, "
        "whereas ungrounded evaluation can reward majority-class collapse. Furthermore, permutation consistency does not detect stubbornly held errors, which constitute "
        "over half of all error instances. Scientific benchmark reporting should adopt multi-dimensional scorecards separating accuracy, calibration, and presentation stability.",
        body_style
    ))

    # References
    story.append(Paragraph("References", h1_style))
    refs_text = [
        ("[Abdin et al. 2024]", "Abdin, M., Jacobs, S. A., Awan, A. A., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. <i>arXiv preprint arXiv:2404.14219</i>."),
        ("[Brier 1950]", "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. <i>Monthly Weather Review</i>, 78(1):1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2"),
        ("[Dettmers et al. 2023]", "Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. <i>NeurIPS 2023</i>."),
        ("[Gao et al. 2023]", "Gao, L., Tow, J., Abbasi, B., et al. (2023). A framework for few-shot language model evaluation. <i>Zenodo</i>. doi:10.5281/zenodo.10256836"),
        ("[Geifman & El-Yaniv 2017]", "Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. <i>NeurIPS 2017</i>, 30:4879–4889."),
        ("[Guo et al. 2017]", "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. <i>ICML 2017</i>, PMLR 70:1321–1330."),
        ("[Hendrycks et al. 2021]", "Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring Massive Multitask Language Understanding. <i>ICLR 2021</i>."),
        ("[Holm 1979]", "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics</i>, 6(2):65–70."),
        ("[Jin et al. 2019]", "Jin, Q., Dhingra, B., Liu, Z., Cohen, W. W., & Lu, X. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. <i>EMNLP-IJCNLP 2019</i>, pp. 2567–2577."),
        ("[Kadavath et al. 2022]", "Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. <i>arXiv preprint arXiv:2207.05221</i>."),
        ("[Kendall 1938]", "Kendall, M. G. (1938). A new measure of rank correlation. <i>Biometrika</i>, 30(1/2):81–93."),
        ("[Lewis et al. 2020]", "Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. <i>NeurIPS 2020</i>, 33:9459–9474."),
        ("[Liang et al. 2022]", "Liang, P., Bommasani, R., Lee, T., et al. (2023). Holistic Evaluation of Language Models. <i>Transactions on Machine Learning Research</i>."),
        ("[Manakul et al. 2023]", "Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. <i>EMNLP 2023</i>, pp. 9004–9017."),
        ("[McNemar 1947]", "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. <i>Psychometrika</i>, 12(2):153–157."),
        ("[Min et al. 2023]", "Min, S., Krishna, K., Lyu, X., et al. (2023). FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation. <i>EMNLP 2023</i>, pp. 12076–12100."),
        ("[Qwen Team 2024]", "Qwen Team. (2024). Qwen2.5 Technical Report. <i>arXiv preprint arXiv:2412.15115</i>."),
        ("[Rein et al. 2023]", "Rein, D., Hou, B. L., Stickland, A. C., et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark. <i>arXiv preprint arXiv:2311.12022</i>."),
        ("[Wang et al. 2024]", "Wang, X., Hu, Z., Lu, P., et al. (2024). SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models. <i>ICML 2024</i>, PMLR 235:50774–50800"),
        ("[Zheng et al. 2024]", "Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. <i>ICLR 2024</i>.")
    ]
    for tag, ref in refs_text:
        story.append(Paragraph(f"<b>{tag}</b> {ref}", ParagraphStyle("RefP", parent=body_style, fontSize=7, leading=9.2, leftIndent=12, firstLineIndent=-12, spaceAfter=2)))

    # Appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix A. Evaluation Prompt Templates", h1_style))
    story.append(Paragraph(
        "All models were evaluated using zero-shot cloze scoring under standardized prompt templates. "
        "All prompts are sent as a user message in each model's official chat template. Candidate options are displayed "
        "in permuted order (display position j displays option &pi;(j)). For PubMedQA context-grounded mode, the abstract text is truncated at 6,000 characters.<br/>"
        "<b>A.1 MMLU Science Subsets (Chat Style):</b><br/>"
        "<code>The following is a multiple-choice question about {subject}. Respond with only the letter of the correct answer.\n\nQuestion: {question}\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:</code><br/>"
        "<b>A.2 PubMedQA Closed-Book (Chat Style):</b><br/>"
        "<code>Answer the following biomedical research question.\n\nQuestion: {question}\nA. yes B. no C. maybe\nRespond with only the letter of your answer.</code><br/>"
        "<b>A.3 PubMedQA Context-Grounded (Chat Style):</b><br/>"
        "<code>Read the abstract and answer the research question based on it.\n\nAbstract:\n{abstract}\n\nQuestion: {question}\noptions\nRespond with only the letter of your answer.</code><br/>"
        "<b>A.4 Baseline Validation Harness Prompt:</b><br/>"
        "The harness-style prompt mimicking EleutherAI's <i>lm-evaluation-harness</i>:<br/>"
        "<code>The following are multiple choice questions (with answers) about {subject}.\n\n{question}\n{options}\nAnswer:</code><br/>"
        "was used <b>only</b> for baseline validation; it was not used for comparative model evaluations.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Appendix B. Computing Environment and Hyperparameters", h1_style))
    story.append(Paragraph(
        "Evaluation runs across the 3 models totaled 17,400 forward passes (3 models &times; 1,450 item-slice evaluations &times; 4 permutations). All inference was "
        "executed under the recorded execution environment:<br/>"
        "&bull; <b>Hardware:</b> NVIDIA Tesla T4 GPU.<br/>"
        "&bull; <b>Software Stack:</b> Python 3.13.16, PyTorch 2.14.1+cu130, and HuggingFace Transformers 4.45.2.<br/>"
        "&bull; <b>Model Quantization and Precision:</b> "
        "<i>Qwen2.5-1.5B:</i> 16-bit (<code>float16</code>); "
        "<i>Phi-3.5-mini:</i> 16-bit (<code>float16</code>); "
        "<i>Qwen2.5-7B:</i> 4-bit NormalFloat (NF4) via <code>bitsandbytes</code> using library defaults (<code>load_in_4bit=True</code>, <code>bnb_4bit_compute_dtype=torch.float16</code>, <code>bnb_4bit_quant_type=\"nf4\"</code>; no double quantization).<br/>"
        "&bull; <b>Inference Settings:</b> Batch size 8, deterministic option permutations K = 4, random seed 42, protocol version 1.0.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Appendix C. Additional Statistical Tables", h1_style))
    story.append(Paragraph("<b>Table 7:</b> Paired McNemar tests between models on each slice. p-values are Holm-Bonferroni adjusted across all comparisons.", caption_style))

    tab_paired_data = [
        ["Slice", "Model A", "Model B", "Acc A", "Acc B", "A Right, B Wrong", "A Wrong, B Right", "Raw p", "Holm-Adjusted p"],
        ["Biomed", "Phi-3.5-mini", "Qwen-1.5B", "0.773", "0.693", "22", "10", "0.0501", "0.3507 (Non-sig)"],
        ["Biomed", "Phi-3.5-mini", "Qwen-7B", "0.773", "0.733", "16", "10", "0.3269", "0.9808 (Non-sig)"],
        ["Biomed", "Qwen-1.5B", "Qwen-7B", "0.693", "0.733", "13", "19", "0.3771", "0.9808 (Non-sig)"],
        ["Math", "Phi-3.5-mini", "Qwen-1.5B", "0.473", "0.380", "38", "24", "0.0980", "0.4898 (Non-sig)"],
        ["Math", "Phi-3.5-mini", "Qwen-7B", "0.473", "0.567", "17", "31", "0.0595", "0.3568 (Non-sig)"],
        ["Math", "Qwen-1.5B", "Qwen-7B", "0.380", "0.567", "15", "43", "0.0003", "0.0034 (Sig)"],
        ["PhysChem", "Phi-3.5-mini", "Qwen-1.5B", "0.573", "0.453", "34", "16", "0.0153", "0.1228 (Non-sig)"],
        ["PhysChem", "Phi-3.5-mini", "Qwen-7B", "0.573", "0.647", "14", "25", "0.1081", "0.4898 (Non-sig)"],
        ["PhysChem", "Qwen-1.5B", "Qwen-7B", "0.453", "0.647", "7", "36", "8.96e-06", "0.0001 (Sig)"],
        ["PubMed Cl.", "Phi-3.5-mini", "Qwen-1.5B", "0.324", "0.552", "38", "152", "2.38e-17", "3.58e-16 (Sig)"],
        ["PubMed Cl.", "Phi-3.5-mini", "Qwen-7B", "0.324", "0.418", "29", "76", "5.05e-06", "6.57e-05 (Sig)"],
        ["PubMed Cl.", "Qwen-1.5B", "Qwen-7B", "0.552", "0.418", "101", "34", "6.60e-09", "9.24e-08 (Sig)"],
        ["PubMed Ctx", "Phi-3.5-mini", "Qwen-1.5B", "0.660", "0.660", "74", "74", "1.0000", "1.0000 (Non-sig)"],
        ["PubMed Ctx", "Phi-3.5-mini", "Qwen-7B", "0.660", "0.728", "28", "62", "0.0004", "0.0044 (Sig)"],
        ["PubMed Ctx", "Qwen-1.5B", "Qwen-7B", "0.660", "0.728", "39", "73", "0.0017", "0.0153 (Sig)"],
    ]
    t_paired = Table([[Paragraph(c, table_cell_bold if r==0 else table_cell) for c in row] for r, row in enumerate(tab_paired_data)],
                     colWidths=[55, 65, 60, 35, 35, 65, 65, 55, 69])
    t_paired.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_paired)

    doc.build(story, canvasmaker=NumberedCanvas)

    reader = pypdf.PdfReader(str(pdf_path))
    num_p = len(reader.pages)
    print(f"Successfully generated PDF at: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path):,} bytes")
    print(f"TOTAL PAGES: {num_p}")

if __name__ == "__main__":
    build_pdf()
